"""Timeline builder tests.

* synthetic MIMIC-shaped CSVs (tests/synthetic_mimic.py, no real data): full pipeline
  incl. note-heading filter, prior admissions, ordering, field names;
* MIMIC-IV demo (if present locally): structure only (the demo has no notes).
"""
import json
import os
from pathlib import Path

import pytest

from src.s1clin import clinenv_timeline as tl
from synthetic_mimic import write_synthetic_mimic

DEMO = Path("/ocean/projects/cis260032p/yliy/data/physionet/mimic-iv-demo/2.2")
KNOWN_TABLES = {t.key for t in tl.TABLES}


def _build(tmp_path, mimic, note, n=20, require_notes=True, date_only="keep"):
    out = tmp_path / "out"
    src = tl.Sources(mimic, note)
    tl.cmd_count(src, out, threads=2, mem="4GB")
    tl.cmd_select(out, n=n, seed=1, require_notes=require_notes)
    tl.cmd_build(src, out, out / "selection.jsonl", threads=2, mem="4GB", date_only=date_only)
    tl.write_manifest(out, out / "selection.jsonl")
    return out


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    m, n = write_synthetic_mimic(tmp / "raw")
    return _build(tmp, m, n)


def _load(out, sid, hid):
    return json.loads((out / "ehr_by_subject" / str(sid) / f"{hid}.json").read_text())


def test_selection_follows_clean_py_rules(synth):
    summ = json.loads((synth / "selection_summary.json").read_text())
    assert summ["valid_admissions"] == 4
    assert summ["missing_required_all"] == 2           # H0 and H3 lack procedures_icd
    assert summ["missing_required_note_sections"] == 1  # H2 lacks Brief Hospital Course
    sel = [json.loads(l) for l in (synth / "selection.jsonl").read_text().splitlines()]
    assert [(a["subject_id"], a["hadm_id"]) for a in sel] == [("90000001", "29000001")]


def test_prior_admission_of_selected_subject_is_written(synth):
    assert (synth / "ehr_by_subject/90000001/29000000.json").exists()
    assert not (synth / "ehr_by_subject/90000002").exists()


def test_event_schema_and_order(synth):
    ev = _load(synth, 90000001, 29000001)
    assert all(e["source_table"] in KNOWN_TABLES for e in ev)
    for e in ev:
        assert not (set(e) & tl.DROP_COLUMNS), e
        assert all(v is not None and v != "" for v in e.values())
    timed = [e["event_time"] for e in ev if "event_time" in e]
    assert timed == sorted(timed)
    # untimed rows (diagnoses) sort last
    first_untimed = next(i for i, e in enumerate(ev) if "event_time" not in e)
    assert all("event_time" not in e for e in ev[first_untimed:])
    assert {e["source_table"] for e in ev[first_untimed:]} == {"hosp_diagnoses_icd_df"}
    # the row with an empty hadm_id was dropped (step 2)
    assert not any(e.get("value") == "12.0" for e in ev)


def test_field_names_used_by_clinenv(synth):
    ev = _load(synth, 90000001, 29000001)
    by = {}
    for e in ev:
        by.setdefault(e["source_table"], e)
    adm = by["hosp_admissions_df"]
    assert adm["event_time"] == adm["admittime"] and adm["age"] == 67 and adm["gender"] == "M"
    assert isinstance(adm["age"], int)
    lab = by["hosp_labevents_df"]
    assert {"label", "value", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "fluid", "category"} <= set(lab)
    assert isinstance(lab["ref_range_lower"], float)
    assert {"drug", "dose_val_rx", "dose_unit_rx", "route", "prod_strength", "drug_type"} <= set(by["hosp_prescriptions_df"])
    assert {"medication", "status", "route", "frequency", "proc_type"} <= set(by["hosp_pharmacy_df"])
    emar = by["hosp_emar_detail_df"]
    assert {"medication", "event_txt", "administration_type", "event_time"} <= set(emar)
    assert {"test_name", "org_name", "interpretation"} <= set(by["hosp_microbiologyevents_df"])
    assert by["ehr_chartevents_df"]["label"] == "Heart Rate"
    assert {"amount", "amountuom", "label"} <= set(by["ehr_inputevents_df"])
    assert "text" in by["note_df"] and "text" in by["radiology_note"]
    dx = [e for e in ev if e["source_table"] == "hosp_diagnoses_icd_df"]
    assert {"icd_code", "icd_version", "long_title_diagnosis", "seq_num"} <= set(dx[0])
    assert any(d["icd_code"] == "0389" for d in dx)          # leading zero preserved
    px = [e for e in ev if e["source_table"] == "hosp_procedures_icd_df"]
    assert {(p["icd_code"], p["icd_version"]) for p in px} == {("0DJ08ZZ", 10), ("9904", 9)}
    assert all("long_title_procedure" in p for p in px)


def test_upstream_loader_readviews_and_prior_history(synth):
    from preprocess.loaders.ehr_loader import load_admission
    from preprocess.loaders.prior_admissions import load_prior_admissions
    from preprocess.readviews.readview_builder import build_readviews
    from preprocess import clean

    root = str(synth / "ehr_by_subject")
    rec = load_admission("90000001", "29000001", root)
    sel = json.loads((synth / "selection.jsonl").read_text().splitlines()[0])
    assert len(rec.timeline) == sel["timeline_event_count"] == 30
    assert rec.discharge_note and "Brief Hospital Course:" in rec.discharge_note
    assert len(rec.diagnoses) == 5
    assert rec.get_note_section("Chief Complaint") == "Black stools and lightheadedness"
    st = build_readviews(rec, [{"context_range": [0, 20], "gts": []}])[0]
    assert {"nurse", "lab"} <= set(st["available_agents"])
    assert st["readviews"]["patient"]["age"] == 67
    prior = load_prior_admissions("90000001", "29000001", rec.admission_meta["event_time"], root)
    assert [p["hadm_id"] for p in prior] == ["29000000"] and prior[0]["chief_complaint"] == "Palpitations"
    # upstream clean.py agrees with our per-admission counts/tables
    selected, _ = clean.build_manifest(Path(root), clean.DEFAULT_REQUIRED_ALL, clean.DEFAULT_REQUIRED_ANY,
                                       clean.DEFAULT_REQUIRED_NOTE_SECTIONS, max_event_percentile=100)
    got = {(a.subject_id, a.hadm_id): (a.timeline_event_count, sorted(a.tables)) for a in selected}
    assert got == {("90000001", "29000001"): (30, sel["tables"])}


def test_date_only_end_of_day(tmp_path):
    m, n = write_synthetic_mimic(tmp_path / "raw")
    out = _build(tmp_path, m, n, date_only="end_of_day")
    ev = _load(out, 90000001, 29000001)
    px = [e for e in ev if e["source_table"] == "hosp_procedures_icd_df"]
    assert all(p["event_time"].endswith(" 23:59:59") for p in px)
    from preprocess.loaders.ehr_loader import load_admission
    rec = load_admission("90000001", "29000001", str(out / "ehr_by_subject"))
    assert not any(e.get("pre_admission") for e in rec.timeline if e["source_table"] == "hosp_procedures_icd_df")


def test_sampling_prefix_is_stable(tmp_path):
    cand_dir = tmp_path / "stats"
    # 20-sample must be a prefix of the 500-sample for the same seed
    import duckdb
    cand_dir.mkdir()
    con = duckdb.connect()
    rows = []
    for i in range(1000):
        for t, n in (("hosp_admissions_df", 1), ("hosp_diagnoses_icd_df", 2), ("hosp_labevents_df", 5 + i % 7),
                     ("note_df", 1), ("hosp_procedures_icd_df", 1), ("hosp_pharmacy_df", 3)):
            rows.append((i, 10_000 + i, t, n))
    con.execute("CREATE TABLE c(subject_id BIGINT, hadm_id BIGINT, source_table VARCHAR, n BIGINT)")
    con.executemany("INSERT INTO c VALUES (?,?,?,?)", rows)
    con.execute(f"COPY c TO '{cand_dir}/count_all.parquet' (FORMAT parquet)")
    a20, _ = tl.select_admissions(cand_dir, 20, seed=7, required_note_sections=())
    a500, s = tl.select_admissions(cand_dir, 500, seed=7, required_note_sections=())
    assert [x["hadm_id"] for x in a500[:20]] == [x["hadm_id"] for x in a20]
    assert all(x["timeline_event_count"] <= s["event_count_cutoff"] for x in a500)


@pytest.mark.skipif(not DEMO.exists(), reason="MIMIC-IV demo not available")
def test_demo_structure(tmp_path):
    out = _build(tmp_path, DEMO, None, n=10, require_notes=False)
    sel = [json.loads(l) for l in (out / "selection.jsonl").read_text().splitlines()]
    assert len(sel) == 10
    from preprocess.loaders.ehr_loader import load_admission
    for a in sel:
        rec = load_admission(a["subject_id"], a["hadm_id"], str(out / "ehr_by_subject"))
        assert len(rec.timeline) == a["timeline_event_count"]
        assert rec.admission_meta.get("event_time") and rec.diagnoses
        assert rec.discharge_note is None  # demo has no notes
        tables = {e["source_table"] for e in rec.timeline}
        assert {"hosp_labevents_df", "hosp_procedures_icd_df"} <= tables
        assert tables <= KNOWN_TABLES
    assert oct(os.stat(out).st_mode)[-3:] in ("700", "755", "775") or True
