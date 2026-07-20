from recipe.pie_grpo.experiments import install_grpo as I


def test_vendored_inferlet_present():
    assert I.WASM.exists() and I.WASM.stat().st_size > 1000
    assert I.MANIFEST.exists()
    assert 'name = "grpo"' in I.MANIFEST.read_text()
