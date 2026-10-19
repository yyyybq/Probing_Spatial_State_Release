"""Optional evaluator environment switches: neutral names and legacy aliases.

Both evaluators read ``PROBE_EVAL_*`` first and fall back to the older ``Q00_*``
names, so wrappers written for the old names keep working. None of these
variables is needed to reproduce the results.
"""

import pytest
import torch
import torch.nn as nn

import vidfm3d.eval_a1_streaming as eval_a1
import vidfm3d.eval_diag as eval_diag

EVALUATORS = [eval_diag, eval_a1]
SWITCHES = (
    ("PROBE_EVAL_REQUIRE_CUDA", "Q00_REQUIRE_CUDA"),
    ("PROBE_EVAL_MODE", "Q00_EVALUATION_MODE"),
    ("PROBE_EVAL_LEGACY_CONSTANTS_COMPAT", "Q00_LEGACY_AS_IS_STRICT_CONSTANTS_COMPAT"),
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for pair in (*eval_diag._SUMMARY_ENV_LABELS, *SWITCHES):
        for name in pair:
            monkeypatch.delenv(name, raising=False)


def test_both_evaluators_read_the_same_variables():
    assert eval_a1._SUMMARY_ENV_LABELS == eval_diag._SUMMARY_ENV_LABELS
    for name, legacy in eval_diag._SUMMARY_ENV_LABELS:
        assert name.startswith("PROBE_EVAL_") and legacy.startswith("Q00_")
        assert name[len("PROBE_EVAL_"):] == legacy[len("Q00_"):]


@pytest.mark.parametrize("module", EVALUATORS)
def test_neutral_name_is_read_first_and_legacy_alias_still_works(module, monkeypatch):
    read = lambda: module._env_with_alias("PROBE_EVAL_CAMPAIGN", "Q00_CAMPAIGN")  # noqa: E731
    assert read() == ("PROBE_EVAL_CAMPAIGN", None)
    monkeypatch.setenv("Q00_CAMPAIGN", "old")
    assert read() == ("Q00_CAMPAIGN", "old")
    monkeypatch.setenv("PROBE_EVAL_CAMPAIGN", "")  # empty counts as unset
    assert read() == ("Q00_CAMPAIGN", "old")
    monkeypatch.setenv("PROBE_EVAL_CAMPAIGN", "new")
    assert read() == ("PROBE_EVAL_CAMPAIGN", "new")


@pytest.mark.parametrize("module", EVALUATORS)
def test_summary_labels_use_the_name_that_was_read(module, monkeypatch):
    assert module._summary_env_labels() == {}
    monkeypatch.setenv("PROBE_EVAL_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("Q00_CONFIG_SHA256", "b" * 64)  # legacy only: legacy summary key
    monkeypatch.setenv("PROBE_EVAL_PROTOCOL", "new")
    monkeypatch.setenv("Q00_PROTOCOL", "old")  # shadowed by the neutral name
    assert module._summary_env_labels() == {
        "probe_eval_manifest_sha256": "a" * 64,
        "q00_config_sha256": "b" * 64,
        "probe_eval_protocol": "new",
    }


def _no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.mark.parametrize("module", EVALUATORS)
@pytest.mark.parametrize("variable", ["PROBE_EVAL_REQUIRE_CUDA", "Q00_REQUIRE_CUDA"])
def test_require_cuda_fails_instead_of_falling_back_to_cpu(module, variable, monkeypatch):
    _no_cuda(monkeypatch)
    monkeypatch.setenv(variable, "1")
    with pytest.raises(RuntimeError, match=f"^{variable}=1 but CUDA is unavailable"):
        module._execution_device()


@pytest.mark.parametrize("module", EVALUATORS)
def test_cpu_fallback_by_default(module, monkeypatch):
    _no_cuda(monkeypatch)
    device, metadata = module._execution_device()
    assert device.type == "cpu" and metadata["execution_device"] == "cpu"
    monkeypatch.setenv("PROBE_EVAL_REQUIRE_CUDA", "0")
    monkeypatch.setenv("Q00_REQUIRE_CUDA", "1")  # shadowed by the neutral name
    assert module._execution_device()[0].type == "cpu"


class _Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conditioning = "full"
        self.head = nn.Linear(2, 2)
        self.constant_context = nn.Parameter(torch.zeros(2))
        self.constant_action = nn.Parameter(torch.zeros(2))


class _Model(nn.Module):
    probe_type = "action_dynamics"

    def __init__(self):
        super().__init__()
        self.probe = _Probe()


def _historical_state_dict():
    state = _Model().state_dict()
    del state["probe.constant_context"], state["probe.constant_action"]
    return state


def test_legacy_constants_compat_is_off_by_default():
    with pytest.raises(RuntimeError, match="Missing key"):
        eval_diag._load_checkpoint_strict_with_legacy_loose_compat(_Model(), _historical_state_dict())


@pytest.mark.parametrize(
    "mode_var,compat_var",
    [
        ("PROBE_EVAL_MODE", "PROBE_EVAL_LEGACY_CONSTANTS_COMPAT"),
        ("Q00_EVALUATION_MODE", "Q00_LEGACY_AS_IS_STRICT_CONSTANTS_COMPAT"),
    ],
)
def test_legacy_constants_compat_opt_in(mode_var, compat_var, monkeypatch):
    monkeypatch.setenv(mode_var, "LEGACY_AS_IS_LOOSE")
    with pytest.raises(RuntimeError, match="Missing key"):  # both switches are required
        eval_diag._load_checkpoint_strict_with_legacy_loose_compat(_Model(), _historical_state_dict())
    monkeypatch.setenv(compat_var, "1")
    model = _Model()
    info = eval_diag._load_checkpoint_strict_with_legacy_loose_compat(model, _historical_state_dict())
    assert info["mode"] == "removed_runtime_only_legacy_parameters"
    assert not hasattr(model.probe, "constant_context")
