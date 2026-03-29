"""Small tree utilities (device-side, JIT-friendly)."""

import collections
import threading

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree

from .types import PyTree, Scalar, Array


def as_int(x: int, dtype=jnp.int32) -> Array:
    return jnp.asarray(x, dtype=dtype)


def as_float(x: float, dtype=jnp.float32) -> Scalar:
    return jnp.asarray(x, dtype=dtype)


def nan_scalar(dtype=jnp.float32) -> Scalar:
    return jnp.asarray(jnp.nan, dtype=dtype)


def cadence(step: jax.Array, k: int) -> jax.Array:
    # k is a Python int (static). If k < 1, disable.
    if k < 1:
        return jnp.asarray(False)
    if k == 1:
        return jnp.asarray(True)
    return (step % k) == 0


def maybe_cast_tree(tree: PyTree, dtype: jnp.dtype) -> PyTree:
    return jtu.tree_map(lambda a: a.astype(dtype), tree)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jtu.tree_map(jnp.add, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    return jtu.tree_map(jnp.subtract, a, b)


def tree_scale(a: PyTree, s: Scalar) -> PyTree:
    return jtu.tree_map(lambda x: x * s, a)


def tree_add_scaled(a: PyTree, s: Scalar, b: PyTree) -> PyTree:
    """Return a + s * b elementwise over a pytree."""
    # JAX XLA will fuse this add+multiply automatically.
    return jtu.tree_map(lambda x, y: x + s * y, a, b)


def tree_zeros_like(a: PyTree) -> PyTree:
    return jtu.tree_map(jnp.zeros_like, a)


def tree_vdot(a: PyTree, b: PyTree) -> jax.Array:
    """Compute dot product of two pytrees (sum of leaf dot products)."""
    dots = jtu.tree_map(lambda x, y: jnp.vdot(x, y), a, b)
    dt = jnp.result_type(*jtu.tree_leaves(dots))
    init = jnp.asarray(0, dtype=dt)
    return jtu.tree_reduce(jnp.add, dots, initializer=init)


def tree_dot_pair(a: PyTree, b: PyTree) -> tuple[Scalar, Scalar]:
    """Return (a dot a, a dot b) for PyTrees in a single traversal."""
    dt = jnp.result_type(*jtu.tree_leaves(a), *jtu.tree_leaves(b))
    init = jnp.zeros((2,), dtype=dt)

    vals = jtu.tree_map(
        lambda x, y: jnp.stack([jnp.vdot(x, x), jnp.vdot(x, y)]), a, b,
    )
    out = jtu.tree_reduce(lambda acc, t: acc + t, vals, initializer=init)
    return out[0], out[1]


def tree_norm2(a: PyTree) -> Scalar:
    """Compute squared L2 norm of a pytree."""
    sq_norms = jtu.tree_map(lambda x: jnp.real(jnp.vdot(x, x)), a)
    dt = jnp.result_type(*jtu.tree_leaves(sq_norms))
    init = jnp.asarray(0, dtype=dt)
    return jtu.tree_reduce(jnp.add, sq_norms, initializer=init)


def tree_norm(a: PyTree) -> Scalar:
    return jnp.sqrt(tree_norm2(a))


def tree_rms(a: PyTree, *, acc_dtype=jnp.float32, eps=0.0) -> Scalar:
    """RMS over all scalar entries in a PyTree: sqrt(mean(x^2)).
    Single tree traversal. No tree_leaves(). JIT-safe.
    """

    # This implementation is already optimal: it fuses sum-of-squares
    # and element-count into a single pass without materializing a flattened view.
    def reducer(acc, x):
        s2, n = acc
        x = jnp.asarray(x, dtype=acc_dtype)
        s2 = s2 + jnp.real(jnp.vdot(x, x))
        n = n + x.size
        return (s2, n)

    init = (jnp.array(0.0, dtype=acc_dtype), jnp.array(0.0, dtype=acc_dtype))
    s2, n = jtu.tree_reduce(reducer, a, initializer=init)
    return jnp.sqrt(s2 / jnp.maximum(n, 1.0) + eps)


def tree_all_finite(a: PyTree) -> jax.Array:
    """True if all leaves are finite. Avoids stacking."""
    # Optimization: tree_reduce with logical_and avoids creating a list/stack of bools.
    is_finite = jtu.tree_map(lambda x: jnp.all(jnp.isfinite(x)), a)
    return jtu.tree_reduce(jnp.logical_and, is_finite, initializer=jnp.array(True))


def flatten_tree(tree: PyTree) -> jax.Array:
    return ravel_pytree(tree)[0]


def flatten_2d_jacobian(tree: PyTree) -> jax.Array:
    """Flatten a PyTree of (B, ...) leaves into a (B, n_params) matrix.

    If the tree is empty, returns a (0, 0) array.
    """
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((0, 0))
    return jax.vmap(lambda _: ravel_pytree(_)[0], in_axes=(0,))(tree)


def flatten_3d_jacobian(tree: PyTree) -> jax.Array:
    flat_jacs = jax.vmap(flatten_2d_jacobian)(tree)
    return flat_jacs.reshape(-1, flat_jacs.shape[-1])


def prefetch_to_device_single(iterator, size: int = 2, device=None):
    """Prefetch batches to a single JAX device in a background thread."""
    if size <= 0:
        raise ValueError(f"size must be >= 1, got {size}")
    if device is None:
        device = jax.devices()[0]

    queue = collections.deque()
    lock = threading.Lock()
    cv = threading.Condition(lock)
    it = iter(iterator)
    state = {"done": False, "err": None, "stop": False}

    def producer():
        try:
            while True:
                with cv:
                    while len(queue) >= size and state["err"] is None and not state["stop"]:
                        cv.wait()
                    if state["stop"] or state["err"] is not None:
                        return

                x = next(it)
                x_dev = jtu.tree_map(lambda a: jax.device_put(a, device), x)

                with cv:
                    if state["stop"] or state["err"] is not None:
                        return
                    queue.append(x_dev)
                    cv.notify_all()
        except StopIteration:
            with cv:
                state["done"] = True
                cv.notify_all()
        except Exception as e:
            with cv:
                state["err"] = e
                cv.notify_all()

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    try:
        while True:
            with cv:
                while not queue and not state["done"] and state["err"] is None:
                    cv.wait()
                if state["err"] is not None:
                    raise state["err"]
                if queue:
                    item = queue.popleft()
                    cv.notify_all()
                    yield item
                elif state["done"]:
                    return
    finally:
        with cv:
            state["stop"] = True
            cv.notify_all()
