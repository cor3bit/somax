import pytest

import jax
import jax.numpy as jnp
import chex

matfree = pytest.importorskip("matfree", reason="MFLanczos tests require matfree")

from somax.estimators import MFLanczos
from somax import utils


def test_mflanczos_extreme_eigs_close_to_truth(make_spd, mv_from_dense, make_block_pytree, keys, tol):
    template, pack, _ = make_block_pytree(shape_a=(50,), shape_b=(30,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=40.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)

    evals = jnp.linalg.eigvalsh(A)
    lam_min_true = evals[0]
    lam_max_true = evals[-1]

    est = MFLanczos(k=32, reortho="none", custom_vjp=False)

    lam_min, lam_max = est.spectrum(template, state=None, mvp=mv, rng=keys())

    # Shapes
    assert jnp.shape(lam_min) == ()
    assert jnp.shape(lam_max) == ()

    # Sanity for SPD
    assert bool(lam_min >= 0.0)
    assert bool(lam_max >= 0.0)
    assert bool(lam_min <= lam_max)

    # Accuracy: extreme eigenvalues should be close
    chex.assert_trees_all_close(lam_min, lam_min_true, **tol)
    chex.assert_trees_all_close(lam_max, lam_max_true, **tol)


def test_mflanczos_error_decreases_with_k(make_spd, mv_from_dense, make_block_pytree, keys):
    template, pack, _ = make_block_pytree(shape_a=(40,), shape_b=(20,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=25.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)
    lam_max_true = jnp.linalg.eigvalsh(A)[-1]

    def approx(k):
        est = MFLanczos(k=k, reortho="none", custom_vjp=False)
        _, lm = est.spectrum(template, state=None, mvp=mv, rng=keys())
        return lm

    lm8 = approx(8)
    lm32 = approx(32)

    err8 = float(jnp.abs(lm8 - lam_max_true))
    err32 = float(jnp.abs(lm32 - lam_max_true))

    assert err32 < 0.85 * err8, f"err8={err8:.4g}, err32={err32:.4g}"


@pytest.mark.parametrize("reortho", ["none", "full"])
def test_mflanczos_reortho_variants_reasonable(make_spd, mv_from_dense, make_block_pytree, keys, reortho):
    template, pack, _ = make_block_pytree(shape_a=(24,), shape_b=(16,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=15.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)
    lam_max_true = jnp.linalg.eigvalsh(A)[-1]

    est = MFLanczos(k=16, reortho=reortho, custom_vjp=False)
    lam_min, lam_max = est.spectrum(template, state=None, mvp=mv, rng=keys())

    assert jnp.isfinite(lam_min) and jnp.isfinite(lam_max)
    assert lam_min > 0

    rel = float(jnp.abs(lam_max - lam_max_true) / (lam_max_true + 1e-12))
    assert rel < 0.25, f"reortho={reortho}, rel={rel:.4g}"


def test_mflanczos_same_rng_reproduces_ritz_values(make_spd, mv_from_dense, make_block_pytree, keys):
    template, pack, _ = make_block_pytree(shape_a=(30,), shape_b=(10,))
    n = utils.flatten_tree(template).shape[0]

    A = make_spd(n, keys(), cond=12.0, eps=1e-6)
    mv = mv_from_dense(A, pack=pack)
    est = MFLanczos(k=16, reortho="none", custom_vjp=False)

    k1 = keys()
    a1 = est.spectrum(template, state=None, mvp=mv, rng=k1)
    a2 = est.spectrum(template, state=None, mvp=mv, rng=k1)

    assert jnp.allclose(jnp.asarray(a1), jnp.asarray(a2))
    assert jnp.all(jnp.isfinite(jnp.asarray(a1)))
