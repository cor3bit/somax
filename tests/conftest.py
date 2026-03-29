import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu
import optax


# ---------------- Global numeric regime ----------------


@pytest.fixture(scope="session", autouse=True)
def set_jax_x64_off():
    """Force default float32 behavior in all tests."""
    # If user env enables x64, we disable it for this phase.
    jax.config.update("jax_enable_x64", False)


@pytest.fixture
def dtype():
    """Single dtype for this phase."""
    return jnp.float32


@pytest.fixture
def tol():
    """Standard tolerances for float32."""
    return dict(rtol=1e-4, atol=5e-5)


# ---------------- PRNG ----------------


@pytest.fixture
def key():
    """Base PRNGKey for tests; split as needed inside tests."""
    return jax.random.PRNGKey(0)


class KeySeq:
    """Stateful splitter for convenience in multi-step tests.

    Keep function-scoped (pytest default). Do not store globally.
    """

    def __init__(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def take(self, n: int = 1):
        ks = jax.random.split(self._key, n + 1)
        self._key = ks[0]
        return ks[1:] if n > 1 else ks[1]

    def __call__(self, n: int = 1):
        return self.take(n)


@pytest.fixture
def keys():
    return KeySeq(0)


# ---------------- PyTree helpers ----------------


@pytest.fixture
def tree_pack_unpack():
    """Return (pack, unpack template) helpers for consistent flattening."""

    def _make(template):
        flat0, pack = ravel_pytree(template)
        return flat0, pack

    return _make


@pytest.fixture
def tree_l2_norm():
    """Compute L2 norm of a PyTree as a JAX scalar (no host sync)."""

    def _norm(tree):
        leaves = jtu.tree_leaves(tree)
        if not leaves:
            return jnp.array(0.0, dtype=jnp.float32)
        flats = [jnp.ravel(jnp.asarray(x)) for x in leaves]
        v = jnp.concatenate(flats, axis=0) if len(flats) > 1 else flats[0]
        return jnp.linalg.norm(v)

    return _norm


@pytest.fixture
def assert_allclose_tree():
    """Allclose for PyTrees with shape checking."""

    def _assert(a, b, *, rtol, atol):
        la, ta = jtu.tree_flatten(a)
        lb, tb = jtu.tree_flatten(b)
        assert ta == tb, "PyTree structure mismatch"
        assert len(la) == len(lb), "PyTree leaf count mismatch"
        for i, (xa, xb) in enumerate(zip(la, lb)):
            xa = jnp.asarray(xa)
            xb = jnp.asarray(xb)
            assert xa.shape == xb.shape, f"Leaf {i} shape mismatch: {xa.shape} vs {xb.shape}"
            np.testing.assert_allclose(np.asarray(xa), np.asarray(xb), rtol=rtol, atol=atol)

    return _assert


# ---------------- Common shapes ----------------


@pytest.fixture
def small_shapes():
    """Tiny shapes for quick unit tests."""
    return dict(B=16, D=5, H=8, C=3)


@pytest.fixture
def medium_shapes():
    """Moderate shapes for smoke tests."""
    return dict(B=64, D=20, H=16, C=4)


# ---------------- Model heads ----------------


@pytest.fixture
def mlp_apply():
    def f(params, x):
        # x: (B, D); params: W1 (D,H), b1 (H,), W2 (H,1), b2 ()
        h = jnp.tanh(x @ params["W1"] + params["b1"])
        y = (h @ params["W2"]).squeeze(-1) + params["b2"]
        return y  # (B,)

    return f


@pytest.fixture
def mlp_logits():
    def f(params, x):
        # x: (B, D); params: W1 (D,H), b1 (H,), W2 (H,C), b2 (C,)
        h = jnp.tanh(x @ params["W1"] + params["b1"])
        return h @ params["W2"] + params["b2"]  # (B, C)

    return f


@pytest.fixture
def linear_predict():
    def f(params, x):
        # params: W (D,1), b (); x: (B,D) -> (B,)
        return (x @ params["W"]).squeeze(-1) + params["b"]

    return f


@pytest.fixture
def linear_logits():
    def f(params, x):
        # params: W (D,C), b (C,)
        return x @ params["W"] + params["b"]

    return f


# ---------------- Losses / metrics ----------------


@pytest.fixture
def mse_loss(mlp_apply):
    def loss(params, batch):
        y_pred = mlp_apply(params, batch["x"])
        y_true = batch["y"]
        if y_true.ndim == 2 and y_true.shape[-1] == 1:
            y_true = y_true.squeeze(-1)
        return 0.5 * jnp.mean((y_pred - y_true) ** 2)

    return loss


@pytest.fixture
def ce_loss(mlp_logits):
    def loss(params, batch):
        logits = mlp_logits(params, batch["x"])
        y = batch["y"]
        if jnp.issubdtype(y.dtype, jnp.integer):
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        # Soft labels (float) path
        y = y.astype(logits.dtype)
        return jnp.mean(optax.softmax_cross_entropy(logits, y))

    return loss


@pytest.fixture
def accuracy(mlp_logits):
    def acc(params, batch):
        logits = mlp_logits(params, batch["x"])
        y = batch["y"]
        # Expect integer labels here; if not, caller should convert.
        return jnp.mean((jnp.argmax(logits, axis=-1) == y).astype(jnp.float32))

    return acc


# ---------------- Param initializers ----------------


@pytest.fixture
def init_mlp_regression(dtype):
    def init(key, D: int, H: int):
        kW1, kb1, kW2, kb2 = jax.random.split(key, 4)
        return {
            "W1": jax.random.normal(kW1, (D, H), dtype=dtype) * 0.05,
            "b1": jax.random.normal(kb1, (H,), dtype=dtype) * 0.01,
            "W2": jax.random.normal(kW2, (H, 1), dtype=dtype) * 0.05,
            "b2": jax.random.normal(kb2, (), dtype=dtype),
        }

    return init


@pytest.fixture
def init_mlp_classification(dtype):
    def init(key, D: int, H: int, C: int):
        kW1, kb1, kW2, kb2 = jax.random.split(key, 4)
        return {
            "W1": jax.random.normal(kW1, (D, H), dtype=dtype) * 0.05,
            "b1": jax.random.normal(kb1, (H,), dtype=dtype) * 0.01,
            "W2": jax.random.normal(kW2, (H, C), dtype=dtype) * 0.05,
            "b2": jax.random.normal(kb2, (C,), dtype=dtype) * 0.01,
        }

    return init


# ---------------- Teacher data generators ----------------


@pytest.fixture
def make_teacher_regression(mlp_apply, dtype):
    def make(key, B: int, D: int, H: int):
        kx, kW1, kb1, kW2, kb2 = jax.random.split(key, 5)
        x = jax.random.normal(kx, (B, D), dtype=dtype)
        teacher = {
            "W1": jax.random.normal(kW1, (D, H), dtype=dtype) * 0.8,
            "b1": jax.random.normal(kb1, (H,), dtype=dtype) * 0.1,
            "W2": jax.random.normal(kW2, (H, 1), dtype=dtype) * 0.8,
            "b2": jax.random.normal(kb2, (), dtype=dtype) * 0.1,
        }
        y = mlp_apply(teacher, x)
        return teacher, {"x": x, "y": y}

    return make


@pytest.fixture
def make_teacher_classification(mlp_logits, dtype):
    def make(key, B: int, D: int, H: int, C: int):
        kx, kW1, kb1, kW2, kb2 = jax.random.split(key, 5)
        x = jax.random.normal(kx, (B, D), dtype=dtype)
        teacher = {
            "W1": jax.random.normal(kW1, (D, H), dtype=dtype) * 0.8,
            "b1": jax.random.normal(kb1, (H,), dtype=dtype) * 0.1,
            "W2": jax.random.normal(kW2, (H, C), dtype=dtype) * 0.8,
            "b2": jax.random.normal(kb2, (C,), dtype=dtype) * 0.1,
        }
        logits = mlp_logits(teacher, x)
        y = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return teacher, {"x": x, "y": y}

    return make


# ---------------- CE block-diagonal helper ----------------


@pytest.fixture
def block_diag():
    def f(sample):
        # sample: (B, n, n) -> (B*n, B*n)
        B, n, _ = sample.shape
        eye = jnp.eye(B, dtype=sample.dtype).reshape(B, 1, B, 1)
        sample4 = sample.reshape(B, n, 1, n)
        out = eye * sample4
        return out.reshape(B * n, B * n)

    return f


# ---------------- Linear-algebra helpers ----------------


@pytest.fixture
def make_spd(dtype):
    """Dense SPD matrix with controlled spectrum."""

    def _make_spd(n: int, key, cond: float = 50.0, eps: float = 1e-6):
        M = jax.random.normal(key, (n, n), dtype=dtype)
        Q, _ = jnp.linalg.qr(M)
        evals = jnp.linspace(1.0, cond, n, dtype=dtype)
        A = (Q * evals) @ Q.T
        # Symmetrize to kill QR numerical asymmetry, then add eps I.
        A = 0.5 * (A + A.T)
        return A + eps * jnp.eye(n, dtype=dtype)

    return _make_spd


@pytest.fixture
def mv_from_dense():
    """Turn dense A into a matvec. If pack is given, accepts/returns PyTrees."""

    def _mv(A, pack=None):
        if pack is None:
            def mv(x):
                return A @ x

            return mv

        def mv(x_tree):
            x_flat, _ = ravel_pytree(x_tree)
            y_flat = A @ x_flat
            return pack(y_flat)

        return mv

    return _mv


@pytest.fixture
def make_block_pytree(dtype):
    """Create a small PyTree with two leaves and a matching ravel/pack pair."""

    def _make_block(shape_a=(5,), shape_b=(3,)):
        template = {"a": jnp.zeros(shape_a, dtype), "b": jnp.ones(shape_b, dtype)}
        flat0, pack = ravel_pytree(template)

        def zeros_like():
            return pack(jnp.zeros_like(flat0))

        return template, pack, zeros_like

    return _make_block


# ---------------- Row-space helpers ----------------


@pytest.fixture
def make_row_operator_from_J():
    """Build a RowOperator from a dense J (m x n), for row-lane tests."""

    def _make(
            J: jax.Array,
            rhs_row: jax.Array,
            pack,
            *,
            b: int,
    ):
        # Lazy import to avoid somax imports at test collection time.
        from somax.curvature.base import RowOperator

        J = jnp.asarray(J)
        rhs_row = jnp.asarray(rhs_row)
        m, n = J.shape

        def jvp(delta_tree):
            delta_flat, _ = ravel_pytree(delta_tree)
            if delta_flat.shape[0] != n:
                raise ValueError(f"delta_flat has wrong size: {delta_flat.shape[0]} != {n}")
            return J @ delta_flat  # (m,)

        def vjp(v_row):
            v_row = jnp.asarray(v_row, dtype=J.dtype)
            return pack(J.T @ v_row)

        return RowOperator(
            rhs=rhs_row,
            b=int(b),
            jvp=jvp,
            vjp=vjp,
        )

    return _make


@pytest.fixture
def bind_row_system_from_J(make_row_operator_from_J):
    """Build RowOperator then bind (A_mv, rhs, backproject, mu)."""

    def _bind(J, rhs_row, pack, *, lam, b: int, reduction: str = "mean"):
        from somax.solvers.row_common import bind_row_system
        row_op = make_row_operator_from_J(J, rhs_row, pack, b=b)
        return bind_row_system(row_op, jnp.asarray(lam, dtype=jnp.asarray(J).dtype), reduction=reduction)

    return _bind


@pytest.fixture
def rel_resid_param_space(tree_l2_norm):
    """Compute ||H s - g|| / ||g|| given matvec H_mv (PyTree->PyTree)."""

    def _rel(H_mv, s, g):
        r = jtu.tree_map(lambda a, b: a - b, H_mv(s), g)
        num = tree_l2_norm(r)
        den = jnp.maximum(tree_l2_norm(g), jnp.array(1e-12, dtype=num.dtype))
        return num / den  # JAX scalar

    return _rel


# ---------------- Non-fixture helpers ----------------
# Prefer moving to tests/_linalg.py later, but OK for now.


def make_full_rank_J(m: int, n: int, key, dtype=jnp.float32):
    """Dense J (m x n) with full column rank when m >= n and well-spread spectrum."""
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (m, m), dtype=dtype)
    Q, _ = jnp.linalg.qr(A)
    B = jax.random.normal(k2, (n, n), dtype=dtype)
    R, _ = jnp.linalg.qr(B)
    D = jnp.linspace(0.5, 2.0, min(m, n), dtype=dtype)
    return (Q[:, :n] * D) @ R.T  # (m, n)


def normal_eq_solution(J: jax.Array, rhs_row: jax.Array, mu: float):
    """Solve (J^T J + mu I) s = J^T rhs_row in param space (reference)."""
    n = J.shape[1]
    JT = J.T
    H = JT @ J + mu * jnp.eye(n, dtype=J.dtype)
    g = JT @ rhs_row
    L = jnp.linalg.cholesky(H)
    y = jax.lax.linalg.triangular_solve(L, g, left_side=True, lower=True)
    s = jax.lax.linalg.triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)
    return s
