"""M5D spec section 34: "Compatibility" - supported config, capability
mismatch, unknown runtime, provider unavailable. All network-free: the
Ollama capability query is monkeypatched at the module function.
"""

import src.evaluation.compatibility as compat_module
from src.evaluation.compatibility import CompatibilityStatus, check_compatibility
from src.evaluation.spec import VariantSpec


def test_native_with_mock_provider_is_supported():
    variant = VariantSpec(variant_id="v", agent_runtime="native", provider="mock", model="mock-model")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.SUPPORTED
    assert result.supported is True


def test_unknown_runtime_is_configuration_error():
    variant = VariantSpec(variant_id="v", agent_runtime="langgraph", provider="mock", model="mock-model")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.CONFIGURATION_ERROR
    assert result.supported is False


def test_strands_with_capability_mismatch(monkeypatch):
    # Simulates the real, confirmed finding: qwen:latest supports only
    # "completion" on Ollama, no "tools".
    monkeypatch.setattr(compat_module, "_query_ollama_capabilities", lambda model, base_url, timeout=5.0: (True, ["completion"]))
    variant = VariantSpec(variant_id="v", agent_runtime="strands", provider="ollama", model="qwen:latest")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.MODEL_CAPABILITY_MISMATCH
    assert result.supported is False
    assert "tools" in result.detail["required"]


def test_strands_with_matching_capability_is_supported(monkeypatch):
    monkeypatch.setattr(
        compat_module, "_query_ollama_capabilities", lambda model, base_url, timeout=5.0: (True, ["completion", "tools", "thinking"])
    )
    variant = VariantSpec(variant_id="v", agent_runtime="strands", provider="ollama", model="gemma4:latest")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.SUPPORTED


def test_provider_unavailable_when_server_unreachable(monkeypatch):
    monkeypatch.setattr(compat_module, "_query_ollama_capabilities", lambda model, base_url, timeout=5.0: (False, None))
    variant = VariantSpec(variant_id="v", agent_runtime="strands", provider="ollama", model="gemma4:latest")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.PROVIDER_UNAVAILABLE


def test_model_unavailable_when_server_reachable_but_model_missing(monkeypatch):
    monkeypatch.setattr(compat_module, "_query_ollama_capabilities", lambda model, base_url, timeout=5.0: (True, None))
    variant = VariantSpec(variant_id="v", agent_runtime="strands", provider="ollama", model="does-not-exist:latest")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.MODEL_UNAVAILABLE


def test_strands_runtime_unavailable_when_sdk_missing(monkeypatch):
    import src.agents.strands_agent as strands_module

    def fail_import():
        raise strands_module.StrandsUnavailableError("not installed")

    monkeypatch.setattr(strands_module, "_import_strands", fail_import)
    variant = VariantSpec(variant_id="v", agent_runtime="strands", provider="ollama", model="gemma4:latest")
    result = check_compatibility(variant)
    assert result.status == CompatibilityStatus.RUNTIME_UNAVAILABLE
