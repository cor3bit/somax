import pytest


def test_register_custom_method():
    from somax.presets import register, make, describe

    @register("my_custom_method", desc="A test method")
    def my_factory(a=1):
        return {"type": "custom", "val": a}

    info = describe("my_custom_method")
    assert info.name == "my_custom_method"
    assert "A test method" in info.desc

    obj = make("my_custom_method", a=42)
    assert obj == {"type": "custom", "val": 42}

    with pytest.raises(KeyError):
        @register("my_custom_method")
        def duplicate():
            pass


def test_root_public_surface():
    import somax as sx

    assert isinstance(sx.__version__, str) and len(sx.__version__) > 0

    _ = (sx.assemble, sx.make, sx.list_methods, sx.describe)
    _ = (sx.SecondOrderMethod, sx.SecondOrderState)

    # Optax integration submodule
    from somax.optax import build_optax_tx, sophia_tx  # noqa: F401

    # types are under somax.types
    from somax.types import Params, Updates, Batch, PRNGKey, Scalar, PyTree  # noqa: F401


def test_presets_loaded_on_root_import_and_contains_known_keys():
    import somax

    methods = somax.list_methods()
    assert isinstance(methods, dict)
    assert len(methods) > 0

    # Curated known presets.
    assert "egn_mse" in methods, f"Registered presets: {sorted(methods)}"
    assert "egn_ce" in methods, f"Registered presets: {sorted(methods)}"

    assert isinstance(methods["egn_mse"], str)
    assert isinstance(methods["egn_ce"], str)


def test_make_unknown_raises_keyerror():
    import somax

    with pytest.raises(KeyError):
        somax.make("this_preset_does_not_exist", foo=1)


def test_explicit_method_exists_and_matches_make_target():
    import somax
    from somax import presets as _presets

    assert hasattr(somax, "egn_ce")
    assert callable(somax.egn_ce)

    fn = _presets._REG["egn_ce"]
    assert fn is somax.egn_ce
