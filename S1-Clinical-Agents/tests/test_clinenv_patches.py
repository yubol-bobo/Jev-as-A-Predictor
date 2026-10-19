"""Tests for the local-only / bug-fix patches in third_party/clinenv."""
import json
import socket

import pytest

import env  # noqa: F401  (installs the guard)
from env import local_guard


def test_socket_guard_blocks_non_loopback():
    assert local_guard.is_installed()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(local_guard.LocalOnlyViolation):
        s.connect(("8.8.8.8", 443))
    s.close()


def test_assert_local_url():
    local_guard.assert_local_url("http://127.0.0.1:8011/v1")
    local_guard.assert_local_url("http://localhost:8000/v1")
    for bad in ("https://api.openai.com/v1", "https://api.deepseek.com", None, ""):
        with pytest.raises(local_guard.LocalOnlyViolation):
            local_guard.assert_local_url(bad)


def test_clients_refuse_remote_config(tmp_path, monkeypatch):
    from env import llm_client
    cfg = tmp_path / "eval.json"
    cfg.write_text(json.dumps({"api": {"base_url": "https://api.openai.com/v1"}}))
    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(cfg))
    with pytest.raises(local_guard.LocalOnlyViolation):
        llm_client.get_openai_client(is_env=True)
    cfg.write_text(json.dumps({}))  # missing config must not fall back to OpenAI
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    with pytest.raises(local_guard.LocalOnlyViolation):
        llm_client.get_openai_client(is_env=True)
    with pytest.raises(local_guard.LocalOnlyViolation):
        llm_client.get_anthropic_client()


def test_shipped_configs_are_local():
    from env.config_loader import get_config
    for name in ("planner", "runtime", "eval"):
        c = get_config(name)
        local_guard.assert_local_url(c["api"]["base_url"], name)
    assert get_config("eval")["embedding"]["base_url"] is None  # in-process local model


def test_cloud_keys_removed():
    import os
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        assert k not in os.environ


def test_icd_version_from_record():
    from evaluation.scorer import _icd_vocab
    assert _icd_vocab({"type": "procedure", "icd_code": "0DJ08ZZ", "icd_version": 10}) == "ICD10PROC"
    assert _icd_vocab({"type": "procedure", "icd_code": "4513", "icd_version": 9}) == "ICD9PROC"
    assert _icd_vocab({"type": "diagnosis", "icd_code": "E119", "icd_version": 10}) == "ICD10CM"
    assert _icd_vocab({"type": "diagnosis", "icd_code": "V5861", "icd_version": 9}) == "ICD9CM"
    assert _icd_vocab({"type": "procedure", "icd_code": "0DJ08ZZ", "icd_version": "10"}) == "ICD10PROC"
    # heuristic still used when the version is missing
    assert _icd_vocab({"type": "procedure", "icd_code": "4513"}) == "ICD9PROC"


def test_icd_normalize_and_hdf1():
    pytest.importorskip("pyhealth")
    from evaluation.scorers.icd_scoring import _normalize, hierarchical_f1
    assert _normalize("0DJ08ZZ", "ICD10PROC") == "0DJ08ZZ"
    assert _normalize("4513", "ICD9PROC") == "45.13"
    assert _normalize("E119", "ICD10CM") == "E11.9"
    assert _normalize("E8786", "ICD9CM") == "E878.6"
    try:
        assert hierarchical_f1("0DJ08ZZ", _normalize("0DJ08ZZ", "ICD10PROC"), "ICD10PROC") == 1.0
        assert 0 < hierarchical_f1("0DJ07ZZ", "0DJ08ZZ", "ICD10PROC") < 1.0
    except local_guard.LocalOnlyViolation:
        pytest.skip("pyhealth vocab not cached locally")


def test_offline_atc():
    pytest.importorskip("pyhealth")
    from evaluation.scorers import atc_scoring as a
    try:
        assert a.drug_to_atc("Metoprolol Tartrate") == "C07AB02"
        assert "J01DD04" in a.drug_to_atc_codes("CeftriaXONE")
        assert a.score_medication("metoprolol", "Metoprolol Succinate XL") == 1.0
        assert a.score_medication("atenolol", "metoprolol") == pytest.approx(0.8)  # same L4 C07AB
        assert a._rxcui("anything") is None and a._ndcs("1") == []
    except local_guard.LocalOnlyViolation:
        pytest.skip("pyhealth ATC vocab not cached locally")


def test_phase_a_uses_full_discharge_note():
    from preprocess.loaders.ehr_loader import AdmissionRecord
    import importlib
    pa = importlib.import_module("preprocess.planners.workflow.phase_a")

    note = ("Chief Complaint:\nX\n\nHistory of Present Illness:\nY\n\nBrief Hospital Course:\nZ\n\n"
            "Discharge Medications:\nUNIQUE_MED_SECTION_TOKEN\n\nDischarge Diagnosis:\nD\n")
    rec = AdmissionRecord(subject_id="1", hadm_id="2", discharge_note=note, timeline=[])
    seen = {}

    def fake_call_llm(client, messages, **kw):
        seen["prompt"] = messages[0]["content"]
        msg = type("M", (), {"content": '{"decisions": []}', "tool_calls": None})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

    orig = pa.call_llm
    pa.call_llm = fake_call_llm
    try:
        pa.phase_a(None, rec, verbose=False, model="x")
    finally:
        pa.call_llm = orig
    assert "UNIQUE_MED_SECTION_TOKEN" in seen["prompt"]  # a section outside the 4-section fallback


def test_text_tool_call_parser():
    from preprocess.planners.workflow.helpers import _parse_text_tool_calls
    xml = "<tool_call>\n<function=mark_single>\n<parameter=index>\n12\n</parameter>\n</function>\n</tool_call>"
    calls = _parse_text_tool_calls(xml, {"mark_single"})
    assert len(calls) == 1 and json.loads(calls[0].function.arguments) == {"index": 12}
    js = '<tool_call>{"name": "mark_single", "arguments": {"index": 3}}</tool_call>'
    assert json.loads(_parse_text_tool_calls(js, {"mark_single"})[0].function.arguments) == {"index": 3}
    assert _parse_text_tool_calls('{"found": false}', {"mark_single"}) == []


def test_strip_think():
    from env.llm_client import _strip_think
    assert _strip_think("<think>abc</think>\n{\"a\": 1}") == '{"a": 1}'
    assert _strip_think('{"a": 1}') == '{"a": 1}'
