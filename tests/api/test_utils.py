import pytest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
import chex

from somax import utils


def test_as_int_default_dtype():
    x = utils.as_int(42)
    assert x.dtype == jnp.int32
    assert x.item() == 42


def test_as_int_custom_dtype():
    # Robust check: int64 request might result in int32 if x64 is disabled
    target_dtype = jnp.int64
    expected_dtype = jnp.array(0, dtype=target_dtype).dtype

    x = utils.as_int(100, dtype=target_dtype)
    assert x.dtype == expected_dtype
    assert x.item() == 100


def test_as_float_default_dtype():
    x = utils.as_float(3.14)
    assert x.dtype == jnp.float32
    assert jnp.allclose(x, jnp.array(3.14, dtype=jnp.float32))


def test_as_float_custom_dtype():
    # Robust check: float64 request might result in float32 if x64 is disabled
    target_dtype = jnp.float64
    expected_dtype = jnp.array(0.0, dtype=target_dtype).dtype

    x = utils.as_float(2.71828, dtype=target_dtype)
    assert x.dtype == expected_dtype
    assert jnp.allclose(x, jnp.array(2.71828, dtype=expected_dtype))


def test_nan_scalar_default():
    x = utils.nan_scalar()
    assert x.dtype == jnp.float32
    assert jnp.isnan(x)


def test_nan_scalar_float64():
    target_dtype = jnp.float64
    expected_dtype = jnp.array(0.0, dtype=target_dtype).dtype

    x = utils.nan_scalar(dtype=target_dtype)
    assert x.dtype == expected_dtype
    assert jnp.isnan(x)


def test_cadence_simple():
    assert bool(utils.cadence(jnp.array(0), 5))
    assert not bool(utils.cadence(jnp.array(4), 5))
    assert bool(utils.cadence(jnp.array(5), 5))
    assert bool(utils.cadence(jnp.array(10), 5))


def test_cadence_k_one():
    for i in range(10):
        assert bool(utils.cadence(jnp.array(i), 1))


def test_cadence_k_zero_or_negative():
    assert not bool(utils.cadence(jnp.array(5), 0))
    assert not bool(utils.cadence(jnp.array(7), -1))


@pytest.mark.parametrize("step", [0, 1, 10, 100, 1000])
def test_cadence_jit_compatible(step):
    @jax.jit
    def check(s):
        return utils.cadence(s, 10)

    result = check(jnp.array(step))
    expected = (step % 10) == 0
    assert bool(result) == expected


@pytest.mark.parametrize("k", [0, -3])
def test_cadence_disable_jit(k):
    @jax.jit
    def check(s):
        return utils.cadence(s, k)

    assert not bool(check(jnp.array(0)))
    assert not bool(check(jnp.array(123)))


def test_maybe_cast_tree_simple(key):
    target_dtype = jnp.float64
    expected_dtype = jnp.array(0.0, dtype=target_dtype).dtype

    tree = {"a": jax.random.normal(key, (3,), dtype=jnp.float32)}
    casted = utils.maybe_cast_tree(tree, target_dtype)
    assert casted["a"].dtype == expected_dtype
    assert jnp.allclose(casted["a"], tree["a"])


def test_maybe_cast_tree_nested(key):
    k1, k2 = jax.random.split(key)
    tree = {
        "w": jax.random.normal(k1, (2, 2), dtype=jnp.float32),
        "b": jax.random.normal(k2, (2,), dtype=jnp.float32),
    }
    casted = utils.maybe_cast_tree(tree, jnp.bfloat16)
    assert casted["w"].dtype == jnp.bfloat16
    assert casted["b"].dtype == jnp.bfloat16


@pytest.fixture
def tree_pair(key):
    k1, k2 = jax.random.split(key)
    a = {"x": jax.random.normal(k1, (4,)), "y": jax.random.normal(k1, (2, 3))}
    b = {"x": jax.random.normal(k2, (4,)), "y": jax.random.normal(k2, (2, 3))}
    return a, b


def test_tree_add(tree_pair):
    a, b = tree_pair
    c = utils.tree_add(a, b)
    chex.assert_trees_all_close(c, {"x": a["x"] + b["x"], "y": a["y"] + b["y"]})


def test_tree_sub(tree_pair):
    a, b = tree_pair
    c = utils.tree_sub(a, b)
    chex.assert_trees_all_close(c, {"x": a["x"] - b["x"], "y": a["y"] - b["y"]})


def test_tree_scale(tree_pair):
    a, _ = tree_pair
    s = jnp.array(2.5)
    c = utils.tree_scale(a, s)
    chex.assert_trees_all_close(c, {"x": a["x"] * 2.5, "y": a["y"] * 2.5})


def test_tree_add_scaled(tree_pair):
    a, b = tree_pair
    s = jnp.array(-0.5)
    c = utils.tree_add_scaled(a, s, b)
    chex.assert_trees_all_close(c, {"x": a["x"] - 0.5 * b["x"], "y": a["y"] - 0.5 * b["y"]})


def test_tree_zeros_like(tree_pair):
    a, _ = tree_pair
    z = utils.tree_zeros_like(a)
    chex.assert_trees_all_close(z, {"x": jnp.zeros_like(a["x"]), "y": jnp.zeros_like(a["y"])})


def test_tree_vdot_simple(key):
    k1, k2 = jax.random.split(key)
    a = {"w": jax.random.normal(k1, (5,))}
    b = {"w": jax.random.normal(k2, (5,))}
    result = utils.tree_vdot(a, b)
    expected = jnp.vdot(a["w"], b["w"])
    assert jnp.allclose(result, expected)


def test_tree_vdot_nested(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    a = {"w": jax.random.normal(k1, (3, 4)), "b": jax.random.normal(k2, (4,))}
    b = {"w": jax.random.normal(k3, (3, 4)), "b": jax.random.normal(k4, (4,))}
    result = utils.tree_vdot(a, b)
    expected = jnp.vdot(a["w"].ravel(), b["w"].ravel()) + jnp.vdot(a["b"], b["b"])
    assert jnp.allclose(result, expected)


def test_tree_vdot_complex(key):
    k1, k2 = jax.random.split(key)
    a = {"z": jax.random.normal(k1, (3,)) + 1j * jax.random.normal(k1, (3,))}
    b = {"z": jax.random.normal(k2, (3,)) + 1j * jax.random.normal(k2, (3,))}
    result = utils.tree_vdot(a, b)
    expected = jnp.vdot(a["z"], b["z"])
    assert jnp.allclose(result, expected)


def test_tree_vdot_mismatched_structure(key):
    a = {"w": jax.random.normal(key, (3,))}
    b = {"w": jax.random.normal(key, (3,)), "b": jnp.zeros(2)}
    with pytest.raises(ValueError):
        utils.tree_vdot(a, b)


def test_tree_vdot_jit_compatible(key):
    @jax.jit
    def compute(a, b):
        return utils.tree_vdot(a, b)

    k1, k2 = jax.random.split(key)
    a = {"x": jax.random.normal(k1, (10,))}
    b = {"x": jax.random.normal(k2, (10,))}
    result = compute(a, b)
    assert result.shape == ()


def _make_pytree(key):
    k1, k2, k3 = jax.random.split(key, 3)
    a = {
        "w": jax.random.normal(k1, (7,), dtype=jnp.float32),
        "b": (
            jax.random.normal(k2, (3, 5), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        ),
    }
    b = {
        "w": jax.random.normal(k3, (7,), dtype=jnp.float32),
        "b": (
            jax.random.normal(k2, (3, 5), dtype=jnp.float32),
            jnp.zeros((0,), dtype=jnp.float32),
        ),
    }
    return a, b


def test_tree_dot_pair_matches_tree_vdot(keys):
    a, b = _make_pytree(keys())

    r2, rz = utils.tree_dot_pair(a, b)

    r2_ref = utils.tree_vdot(a, a)
    rz_ref = utils.tree_vdot(a, b)

    assert isinstance(r2, jax.Array)
    assert isinstance(rz, jax.Array)

    assert jnp.allclose(r2, r2_ref, rtol=0.0, atol=0.0)
    assert jnp.allclose(rz, rz_ref, rtol=0.0, atol=0.0)


def test_tree_dot_pair_consistent_with_tree_norm2(keys):
    a, _ = _make_pytree(keys())

    r2, _ = utils.tree_dot_pair(a, a)
    norm2 = utils.tree_norm2(a)

    assert jnp.allclose(r2, norm2, rtol=0.0, atol=0.0)


def test_tree_dot_pair_jit_safe(keys):
    a, b = _make_pytree(keys())

    f = jax.jit(utils.tree_dot_pair)

    r2_1, rz_1 = f(a, b)
    r2_2, rz_2 = f(a, b)

    assert jnp.allclose(r2_1, r2_2, rtol=0.0, atol=0.0)
    assert jnp.allclose(rz_1, rz_2, rtol=0.0, atol=0.0)


def test_tree_dot_pair_linear_in_second_argument(keys):
    a, b = _make_pytree(keys())
    c = jtu.tree_map(lambda x: 0.37 * x, b)

    r2, rz_b = utils.tree_dot_pair(a, b)
    _r2, rz_c = utils.tree_dot_pair(a, c)

    assert jnp.allclose(r2, _r2, rtol=0.0, atol=0.0)
    assert jnp.allclose(rz_c, 0.37 * rz_b, rtol=1e-6, atol=1e-6)


def test_tree_norm2_simple():
    a = {"w": jnp.array([3.0, 4.0])}
    result = utils.tree_norm2(a)
    expected = 9.0 + 16.0
    assert jnp.allclose(result, expected)


def test_tree_norm2_nested():
    a = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([1.0])}
    result = utils.tree_norm2(a)
    expected = 1.0 + 4.0 + 9.0 + 16.0 + 1.0
    assert jnp.allclose(result, expected)


def test_tree_norm2_complex():
    a = {"z": jnp.array([1.0 + 2.0j, 3.0 + 4.0j])}
    result = utils.tree_norm2(a)
    expected = 5.0 + 25.0
    assert jnp.allclose(result, expected)


def test_tree_norm():
    a = {"w": jnp.array([3.0, 4.0])}
    result = utils.tree_norm(a)
    expected = 5.0
    assert jnp.allclose(result, expected)


def test_tree_norm_consistency(key):
    k1 = jax.random.split(key)[0]
    a = {"w": jax.random.normal(k1, (10,)), "b": jax.random.normal(k1, (5,))}
    norm2 = utils.tree_norm2(a)
    norm = utils.tree_norm(a)
    assert jnp.allclose(norm, jnp.sqrt(norm2))


def test_tree_rms_uniform():
    a = {"w": jnp.ones((4,)) * 2.0}
    result = utils.tree_rms(a)
    expected = 2.0
    assert jnp.allclose(result, expected)


def test_tree_rms_nested():
    a = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([5.0])}
    result = utils.tree_rms(a)
    expected = jnp.sqrt(11.0)
    assert jnp.allclose(result, expected)


def test_tree_rms_with_eps():
    a = {"w": jnp.zeros((3,))}
    result = utils.tree_rms(a, eps=1e-8)
    expected = jnp.sqrt(1e-8)
    assert jnp.allclose(result, expected)


def test_tree_rms_acc_dtype(key):
    a = {"w": jax.random.normal(key, (100,), dtype=jnp.bfloat16)}
    result = utils.tree_rms(a, acc_dtype=jnp.float32)
    assert result.dtype == jnp.float32


def test_tree_rms_jit_compatible(key):
    @jax.jit
    def compute(a):
        return utils.tree_rms(a)

    a = {"w": jax.random.normal(key, (50,))}
    result = compute(a)
    assert result.shape == ()


def test_all_finite_true(key):
    a = {"w": jax.random.normal(key, (5,)), "b": jnp.ones((3,))}
    result = utils.tree_all_finite(a)
    # Correct: check truthiness, not identity (is True)
    assert result


def test_all_finite_with_nan():
    a = {"w": jnp.array([1.0, jnp.nan, 3.0]), "b": jnp.ones((2,))}
    result = utils.tree_all_finite(a)
    assert not result


def test_all_finite_with_inf():
    a = {"w": jnp.array([1.0, jnp.inf]), "b": jnp.ones((2,))}
    result = utils.tree_all_finite(a)
    assert not result


def test_all_finite_returns_jax_array(key):
    a = {"w": jax.random.normal(key, (3,))}
    result = utils.tree_all_finite(a)
    assert isinstance(result, jax.Array)
    assert result.shape == ()


def test_all_finite_jit_compatible(key):
    @jax.jit
    def check(a):
        return utils.tree_all_finite(a)

    a = {"w": jax.random.normal(key, (10,))}
    result = check(a)
    assert result


def test_all_finite_empty_tree():
    result = utils.tree_all_finite({})
    assert result


def test_flatten_tree_simple():
    tree = {"w": jnp.array([1.0, 2.0, 3.0])}
    flat = utils.flatten_tree(tree)
    assert flat.shape == (3,)
    assert jnp.allclose(flat, jnp.array([1.0, 2.0, 3.0]))


def test_flatten_tree_nested():
    tree = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([5.0])}
    flat = utils.flatten_tree(tree)
    expected, _ = ravel_pytree(tree)
    assert jnp.allclose(flat, expected)


def test_flatten_tree_consistency_with_ravel(key):
    k1, k2 = jax.random.split(key)
    tree = {"w": jax.random.normal(k1, (3, 2)), "b": jax.random.normal(k2, (1,))}
    flat_utils = utils.flatten_tree(tree)
    flat_ref, _ = ravel_pytree(tree)
    assert jnp.allclose(flat_utils, flat_ref)


def test_flatten_2d_jacobian_single_leaf(key):
    jac = {"w": jax.random.normal(key, (4, 5, 3))}
    flat = utils.flatten_2d_jacobian(jac)
    assert flat.shape == (4, 15)


def test_flatten_2d_jacobian_multi_leaf(key):
    k1, k2 = jax.random.split(key)
    jac = {
        "w": jax.random.normal(k1, (8, 2, 2)),
        "b": jax.random.normal(k2, (8, 1, 4)),
    }
    flat = utils.flatten_2d_jacobian(jac)
    assert flat.shape == (8, 8)


def test_flatten_2d_jacobian_empty():
    jac = {}
    flat = utils.flatten_2d_jacobian(jac)
    assert flat.shape == (0, 0)


def test_flatten_3d_jacobian_single_leaf(key):
    jac = {"w": jax.random.normal(key, (4, 3, 5, 2))}
    flat = utils.flatten_3d_jacobian(jac)
    assert flat.shape == (12, 10)


def test_flatten_3d_jacobian_multi_leaf(key):
    k1, k2 = jax.random.split(key)
    jac = {
        "w": jax.random.normal(k1, (2, 5, 2, 2)),
        "b": jax.random.normal(k2, (2, 5, 1, 4)),
    }
    flat = utils.flatten_3d_jacobian(jac)
    assert flat.shape == (10, 8)


def test_flatten_3d_jacobian_consistency():
    # jac shape: (2, 3, 4, 5)
    jac = {"w": jnp.arange(2 * 3 * 4 * 5, dtype=jnp.float32).reshape(2, 3, 4, 5)}

    # 3D flatten: merges (dim0, dim1) -> (6, 20)
    flat_utils = utils.flatten_3d_jacobian(jac)

    # 2D flatten: keeps dim0, flattens rest -> (2, 60)
    flat_2d = utils.flatten_2d_jacobian(jac["w"])

    # Reshape 2D to match 3D structure for comparison
    # (2, 60) -> (2, 3, 20) -> (6, 20)
    flat_reshaped = flat_2d.reshape(2, 3, 20).reshape(6, 20)

    assert jnp.allclose(flat_utils, flat_reshaped)


def test_prefetch_basic():
    def data_gen():
        for i in range(5):
            yield {"x": jnp.array([i], dtype=jnp.float32)}

    prefetched = utils.prefetch_to_device_single(data_gen(), size=2)
    results = list(prefetched)
    assert len(results) == 5
    for i, batch in enumerate(results):
        assert jnp.allclose(batch["x"], jnp.array([i], dtype=jnp.float32))


def test_prefetch_empty_iterator():
    def data_gen():
        if False:
            yield None

    prefetched = utils.prefetch_to_device_single(data_gen(), size=2)
    results = list(prefetched)
    assert len(results) == 0


def test_prefetch_size_validation():
    with pytest.raises(ValueError, match="size must be >= 1"):
        list(utils.prefetch_to_device_single(iter([]), size=0))


def test_prefetch_exception_propagation():
    def data_gen():
        yield {"x": jnp.array([0], dtype=jnp.float32)}
        raise RuntimeError("Test error")

    prefetched = utils.prefetch_to_device_single(data_gen(), size=1)
    batches = []
    with pytest.raises(RuntimeError, match="Test error"):
        for batch in prefetched:
            batches.append(batch)
    assert len(batches) >= 1


def test_all_tree_ops_jittable(key):
    k1, k2 = jax.random.split(key)
    a = {"w": jax.random.normal(k1, (5,))}
    b = {"w": jax.random.normal(k2, (5,))}
    s = jnp.array(2.0)

    @jax.jit
    def compute_all(a, b, s):
        return (
            utils.tree_add(a, b),
            utils.tree_sub(a, b),
            utils.tree_scale(a, s),
            utils.tree_add_scaled(a, s, b),
            utils.tree_zeros_like(a),
            utils.tree_vdot(a, b),
            utils.tree_norm2(a),
            utils.tree_norm(a),
            utils.tree_rms(a),
            utils.tree_all_finite(a),
        )

    results = compute_all(a, b, s)
    assert len(results) == 10


def test_flatten_ops_jittable(key):
    tree = {"w": jax.random.normal(key, (3, 4, 5))}

    @jax.jit
    def compute(tree):
        flat = utils.flatten_tree(tree)
        flat2d = utils.flatten_2d_jacobian(tree)
        flat3d = utils.flatten_3d_jacobian(tree)
        return flat, flat2d, flat3d

    flat, flat2d, flat3d = compute(tree)
    assert flat.shape == (60,)
    assert flat2d.shape == (3, 20)
    assert flat3d.shape == (12, 5)


def test_tree_all_finite_no_host_transfer(key):
    a = {"w": jax.random.normal(key, (100,))}

    @jax.jit
    def check_and_branch(tree):
        is_finite = utils.tree_all_finite(tree)
        return jax.lax.cond(
            is_finite,
            lambda: jnp.array(1.0),
            lambda: jnp.array(0.0),
        )

    result = check_and_branch(a)
    assert result == 1.0
