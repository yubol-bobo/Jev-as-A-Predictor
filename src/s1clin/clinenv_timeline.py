"""ClinEnv admission-timeline builder (reimplementation of paper App. B.2).

Upstream ClinEnv (third_party/clinenv) reads one JSON array per admission from
``ehr_by_subject/{subject_id}/{hadm_id}.json`` but does not ship the code that
builds those files. This module rebuilds them from raw MIMIC-IV v3.1 (+ Note 2.2)
CSVs, following App. B.2's five deterministic steps:

  1. discover the structured and note CSV files and stream them (DuckDB scans the
     .csv.gz files in a streaming fashion; nothing is decompressed to disk);
  2. normalise subject_id / hadm_id and drop rows missing either key;
  3. canonical ``event_time`` = first non-null of
     charttime, starttime, admittime, chartdate, stoptime, endtime;
  4. serialise each row as an event ``{"source_table", "event_time"?, ...payload}``,
     dropping identifier / provider columns and null fields;
  5. group by admission and sort by (event_time [nulls last], source_table,
     original row order) for deterministic tie-breaking.

and only for a SUBSET of admissions, chosen with ``preprocess/clean.py``'s rule:
required tables, required discharge-note headings, and timeline size at or below
the 50th percentile of all eligible admissions (the percentile is computed over
the whole eligible population from per-admission counts, so the subset is drawn
from exactly the population clean.py would keep).

Commands (see docs/clinenv_pipeline.md):
  count   per-admission per-table row counts + note-heading flags (one pass over all CSVs)
  select  apply clean.py's criteria, seeded random sample -> selection.jsonl
  build   extract rows for the selected subjects and write timeline JSONs
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

# --------------------------------------------------------------------------- #
# Table specifications                                                          #
# --------------------------------------------------------------------------- #

EVENT_TIME_PRIORITY = ("charttime", "starttime", "admittime", "chartdate", "stoptime", "endtime")

# Identifier / provider / internal columns removed from event payloads (step 4).
DROP_COLUMNS = {
    "subject_id", "hadm_id", "stay_id", "caregiver_id", "order_provider_id",
    "admit_provider_id", "enter_provider_id", "labevent_id", "specimen_id",
    "microevent_id", "micro_specimen_id", "pharmacy_id", "poe_id", "poe_seq",
    "emar_id", "emar_seq", "note_id", "orderid", "linkorderid",
    # patients-table helper columns (replaced by age)
    "anchor_age", "anchor_year", "anchor_year_group",
}

INT_COLUMNS = {
    "icd_version", "seq_num", "itemid", "hospital_expire_flag", "test_seq", "isolate_num",
    "spec_itemid", "test_itemid", "org_itemid", "ab_itemid", "note_seq", "isopenbag",
    "continueinnextdept", "age",
}
FLOAT_COLUMNS = {
    "valuenum", "ref_range_lower", "ref_range_upper", "amount", "rate", "totalamount",
    "originalamount", "originalrate", "patientweight", "doses_per_24_hrs",
    "dilution_value", "lownormalvalue", "highnormalvalue",
}

# Metadata tables (never enter the clinical timeline; see ehr_loader._METADATA_TABLES)
METADATA_TABLES = ("hosp_admissions_df", "hosp_diagnoses_icd_df", "note_df")


@dataclass
class TableSpec:
    key: str                      # ClinEnv source_table name
    file: str                     # path relative to its root
    root: str = "mimic"           # "mimic" or "note"
    join_sql: str = ""            # extra FROM/JOIN clause after "src"
    extra_select: str = ""        # extra columns from joined tables
    float_value: bool = False     # cast `value` to DOUBLE (numeric-valued ICU tables)
    order_keys: tuple = ()        # natural-key tie-break (approximates original row order)
    hadm_expr: str = "src.hadm_id"
    subject_expr: str = "src.subject_id"
    exclude_src: tuple = ()       # src columns not to select (replaced by joined ones)


TABLES: list[TableSpec] = [
    TableSpec("hosp_admissions_df", "hosp/admissions.csv.gz",
              join_sql="LEFT JOIN {patients} p ON p.subject_id = src.subject_id",
              extra_select=("p.gender AS gender, "
                            "TRY_CAST(p.anchor_age AS INTEGER) + (year(TRY_CAST(src.admittime AS TIMESTAMP))"
                            " - TRY_CAST(p.anchor_year AS INTEGER)) AS age, p.dod AS dod"),
              order_keys=("admittime",)),
    TableSpec("hosp_diagnoses_icd_df", "hosp/diagnoses_icd.csv.gz",
              join_sql=("LEFT JOIN {d_icd_diagnoses} d ON d.icd_code = src.icd_code "
                        "AND d.icd_version = src.icd_version"),
              extra_select="d.long_title AS long_title_diagnosis",
              order_keys=("seq_num_i",)),
    TableSpec("hosp_procedures_icd_df", "hosp/procedures_icd.csv.gz",
              join_sql=("LEFT JOIN {d_icd_procedures} d ON d.icd_code = src.icd_code "
                        "AND d.icd_version = src.icd_version"),
              extra_select="d.long_title AS long_title_procedure",
              order_keys=("seq_num_i",)),
    TableSpec("hosp_labevents_df", "hosp/labevents.csv.gz",
              join_sql="LEFT JOIN {d_labitems} d ON d.itemid = src.itemid",
              extra_select="d.label AS label, d.fluid AS fluid, d.category AS category",
              order_keys=("labevent_id_i",)),
    TableSpec("hosp_microbiologyevents_df", "hosp/microbiologyevents.csv.gz",
              order_keys=("microevent_id_i",)),
    TableSpec("hosp_prescriptions_df", "hosp/prescriptions.csv.gz",
              order_keys=("pharmacy_id_i", "poe_seq_i", "drug_type", "drug")),
    TableSpec("hosp_pharmacy_df", "hosp/pharmacy.csv.gz",
              order_keys=("pharmacy_id_i",)),
    # emar_detail has no hadm_id/charttime/medication: join its parent emar row.
    TableSpec("hosp_emar_detail_df", "hosp/emar_detail.csv.gz",
              join_sql="JOIN {emar} e ON e.emar_id = src.emar_id",
              extra_select=("e.hadm_id AS hadm_id, e.charttime AS charttime, e.medication AS medication, "
                            "e.event_txt AS event_txt, e.scheduletime AS scheduletime, "
                            "e.storetime AS storetime"),
              hadm_expr="e.hadm_id",
              order_keys=("emar_seq_i", "parent_field_ordinal")),
    TableSpec("ehr_chartevents_df", "icu/chartevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label",
              order_keys=("stay_id_i", "itemid_i", "value")),
    TableSpec("ehr_datetime_events_df", "icu/datetimeevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label",
              order_keys=("stay_id_i", "itemid_i")),
    TableSpec("ehr_inputevents_df", "icu/inputevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label",
              order_keys=("orderid_i", "itemid_i")),
    TableSpec("ehr_outputevents_df", "icu/outputevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label", float_value=True,
              order_keys=("stay_id_i", "itemid_i")),
    TableSpec("ehr_ingredientevents_df", "icu/ingredientevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label",
              order_keys=("orderid_i", "itemid_i")),
    TableSpec("ehr_procedureevents_df", "icu/procedureevents.csv.gz",
              join_sql="LEFT JOIN {d_items} d ON d.itemid = src.itemid",
              extra_select="d.label AS label", float_value=True,
              order_keys=("orderid_i", "itemid_i")),
    TableSpec("note_df", "note/discharge.csv.gz", root="note",
              order_keys=("note_seq_i",)),
    TableSpec("radiology_note", "note/radiology.csv.gz", root="note",
              order_keys=("note_seq_i",)),
]
TABLE_BY_KEY = {t.key: t for t in TABLES}

DICT_TABLES = {
    "patients": ("mimic", "hosp/patients.csv.gz"),
    "d_icd_diagnoses": ("mimic", "hosp/d_icd_diagnoses.csv.gz"),
    "d_icd_procedures": ("mimic", "hosp/d_icd_procedures.csv.gz"),
    "d_labitems": ("mimic", "hosp/d_labitems.csv.gz"),
    "d_items": ("mimic", "icu/d_items.csv.gz"),
    "emar": ("mimic", "hosp/emar.csv.gz"),
}

# clean.py defaults (third_party/clinenv/preprocess/clean.py:40-64)
REQUIRED_ALL = ("hosp_admissions_df", "hosp_diagnoses_icd_df", "hosp_labevents_df",
                "note_df", "hosp_procedures_icd_df")
REQUIRED_ANY = ("hosp_prescriptions_df", "hosp_pharmacy_df")
# Same regexes as clean.NOTE_SECTION_PATTERNS, in RE2 syntax for DuckDB.
NOTE_SECTION_REGEX = {
    "chief_complaint": r"(?im)^\s*Chief Complaint:\s*$",
    "hpi": r"(?im)^\s*(History of Present Illness|HPI):\s*$",
    "brief_hospital_course": r"(?im)^\s*Brief Hospital Course:\s*$",
    "discharge_diagnosis": r"(?im)^\s*Discharge Diagnosis:\s*$",
}


# --------------------------------------------------------------------------- #
# Paths / DuckDB helpers                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Sources:
    mimic_dir: Path                 # contains hosp/ and icu/
    note_dir: Path | None = None    # contains note/ (MIMIC-IV-Note)
    present: dict = field(default_factory=dict)

    def path(self, root: str, rel: str) -> Path | None:
        base = self.mimic_dir if root == "mimic" else self.note_dir
        if base is None:
            return None
        p = base / rel
        return p if p.exists() else None


def _csv(path: Path) -> str:
    # all_varchar: keep ICD codes such as '0389' and free text exactly as written.
    return (f"read_csv('{path}', header=true, all_varchar=true, compression='gzip', "
            f"quote='\"', escape='\"', max_line_size=16777216, sample_size=-1)")


def _connect(threads: int, mem: str, tmp: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def _register_dicts(con, src: Sources, needed: set[str]) -> None:
    for name in needed:
        root, rel = DICT_TABLES[name]
        p = src.path(root, rel)
        if p is None:
            raise FileNotFoundError(f"missing dictionary table {rel} under {src.mimic_dir}")
        if name == "emar":
            # only the columns we need, materialised once (emar is large in full MIMIC)
            con.execute(f"CREATE OR REPLACE TEMP TABLE emar AS SELECT emar_id, subject_id, hadm_id, "
                        f"charttime, medication, event_txt, scheduletime, storetime FROM {_csv(p)} "
                        f"WHERE hadm_id IS NOT NULL AND hadm_id <> ''")
        else:
            con.execute(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {_csv(p)}")


def _needed_dicts(spec: TableSpec) -> set[str]:
    return {n for n in DICT_TABLES if "{" + n + "}" in spec.join_sql}


def _from_clause(spec: TableSpec, path: Path, for_count: bool = False) -> str:
    # counting only needs the join when the admission key lives in the joined table (emar)
    if for_count and not spec.hadm_expr.startswith("e."):
        return f"{_csv(path)} src"
    join = spec.join_sql.format(**{n: n for n in DICT_TABLES})
    return f"{_csv(path)} src {join}"


def _key_filter(spec: TableSpec) -> str:
    # step 2: drop rows missing either key ('' counts as missing in all_varchar CSVs)
    return (f"{spec.hadm_expr} IS NOT NULL AND trim({spec.hadm_expr}) <> '' "
            f"AND {spec.subject_expr} IS NOT NULL AND trim({spec.subject_expr}) <> ''")


# --------------------------------------------------------------------------- #
# count: per-admission per-table row counts + note heading flags                #
# --------------------------------------------------------------------------- #

def cmd_count(src: Sources, out_dir: Path, threads: int, mem: str) -> Path:
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = _connect(threads, mem, tmp)
    log = {"tables": {}, "missing": []}
    t_all = time.time()
    parts = []
    for spec in TABLES:
        p = src.path(spec.root, spec.file)
        if p is None:
            log["missing"].append(spec.file)
            print(f"[count] {spec.key}: MISSING {spec.file}", flush=True)
            continue
        if spec.hadm_expr.startswith("e."):
            _register_dicts(con, src, _needed_dicts(spec))
        t0 = time.time()
        out = stats_dir / f"count_{spec.key}.parquet"
        con.execute(f"""
            COPY (
              SELECT TRY_CAST({spec.subject_expr} AS BIGINT) AS subject_id,
                     TRY_CAST({spec.hadm_expr} AS BIGINT) AS hadm_id,
                     '{spec.key}' AS source_table, count(*) AS n
              FROM {_from_clause(spec, p, for_count=True)}
              WHERE {_key_filter(spec)}
              GROUP BY 1, 2
            ) TO '{out}' (FORMAT parquet)""")
        dt = time.time() - t0
        n = con.execute(f"SELECT count(*), coalesce(sum(n),0) FROM '{out}'").fetchone()
        log["tables"][spec.key] = {"admissions": n[0], "rows": int(n[1]), "seconds": round(dt, 1)}
        print(f"[count] {spec.key}: {n[0]} admissions, {int(n[1])} rows, {dt:.1f}s", flush=True)
        parts.append(out)

    # note headings, on the longest discharge note per admission (clean._note_text_for_events)
    p = src.path("note", "note/discharge.csv.gz")
    if p is not None:
        t0 = time.time()
        flags = ", ".join(f"regexp_matches(txt, '{rx}') AS {name}"
                          for name, rx in NOTE_SECTION_REGEX.items())
        out = stats_dir / "note_sections.parquet"
        con.execute(f"""
            COPY (
              SELECT subject_id, hadm_id, {flags}
              FROM (SELECT TRY_CAST(subject_id AS BIGINT) AS subject_id,
                           TRY_CAST(hadm_id AS BIGINT) AS hadm_id,
                           arg_max(text, length(text)) AS txt
                    FROM {_csv(p)}
                    WHERE hadm_id IS NOT NULL AND trim(hadm_id) <> '' AND text IS NOT NULL AND text <> ''
                    GROUP BY 1, 2)
            ) TO '{out}' (FORMAT parquet)""")
        log["note_sections_seconds"] = round(time.time() - t0, 1)
        print(f"[count] note sections: {time.time() - t0:.1f}s", flush=True)
    else:
        log["missing"].append("note/discharge.csv.gz (note headings)")

    log["seconds_total"] = round(time.time() - t_all, 1)
    (stats_dir / "count_log.json").write_text(json.dumps(log, indent=2))
    return stats_dir


# --------------------------------------------------------------------------- #
# select: clean.py criteria + seeded sample                                    #
# --------------------------------------------------------------------------- #

def select_admissions(stats_dir: Path, n: int, seed: int, max_event_percentile: int = 50,
                      required_note_sections=tuple(NOTE_SECTION_REGEX),
                      require_notes: bool = True) -> tuple[list[dict], dict]:
    con = duckdb.connect()
    counts = sorted(stats_dir.glob("count_*.parquet"))
    if not counts:
        raise FileNotFoundError(f"no count_*.parquet in {stats_dir}; run `count` first")
    files = ", ".join(f"'{c}'" for c in counts)
    con.execute(f"CREATE TEMP VIEW c AS SELECT * FROM read_parquet([{files}])")
    rows = con.execute("""
        SELECT subject_id, hadm_id, list(source_table) AS tables,
               sum(n) AS event_count,
               sum(CASE WHEN source_table IN ('hosp_admissions_df','hosp_diagnoses_icd_df','note_df')
                        THEN 0 ELSE n END) AS timeline_event_count
        FROM c GROUP BY 1, 2""").fetchall()
    note_ok: dict[int, bool] = {}
    ns = stats_dir / "note_sections.parquet"
    if ns.exists() and required_note_sections:
        cols = " AND ".join(required_note_sections)
        for hadm, ok in con.execute(f"SELECT hadm_id, ({cols}) FROM '{ns}'").fetchall():
            note_ok[hadm] = bool(ok)

    stats = {"valid_admissions": 0, "missing_required_all": 0, "missing_required_any": 0,
             "missing_required_note_sections": 0, "removed_by_event_percentile": 0}
    required_all = REQUIRED_ALL if require_notes else tuple(t for t in REQUIRED_ALL if t != "note_df")
    cand = []
    for subject_id, hadm_id, tables, ev, tl in rows:
        tables = set(tables)
        if "hosp_admissions_df" not in tables:
            continue  # orphan rows (hadm_id not in admissions) are not admissions
        stats["valid_admissions"] += 1
        if not set(required_all) <= tables:
            stats["missing_required_all"] += 1
            continue
        if not tables & set(REQUIRED_ANY):
            stats["missing_required_any"] += 1
            continue
        if require_notes and required_note_sections and not note_ok.get(hadm_id, False):
            stats["missing_required_note_sections"] += 1
            continue
        cand.append({"subject_id": str(subject_id), "hadm_id": str(hadm_id), "tables": sorted(tables),
                     "event_count": int(ev), "timeline_event_count": int(tl)})

    # percentile filter, identical arithmetic to clean.build_manifest
    if cand and 0 < max_event_percentile < 100:
        cs = sorted(a["timeline_event_count"] for a in cand)
        cutoff = cs[max(int(len(cs) * max_event_percentile / 100) - 1, 0)]
        kept = [a for a in cand if a["timeline_event_count"] <= cutoff]
        stats["removed_by_event_percentile"] = len(cand) - len(kept)
        stats["event_count_cutoff"] = cutoff
        cand = kept
    stats["eligible_admissions"] = len(cand)

    # Seeded permutation; taking a prefix makes the n=20 sample a subset of the n=500 one.
    cand.sort(key=lambda a: (int(a["subject_id"]), int(a["hadm_id"])))
    rng = random.Random(seed)
    order = list(range(len(cand)))
    rng.shuffle(order)
    chosen = [cand[i] for i in order[: n if n > 0 else len(cand)]]
    stats.update({"seed": seed, "n_requested": n, "n_selected": len(chosen),
                  "max_event_percentile": max_event_percentile, "require_notes": require_notes})
    return chosen, stats


def cmd_select(out_dir: Path, n: int, seed: int, require_notes: bool) -> Path:
    chosen, stats = select_admissions(out_dir / "stats", n, seed, require_notes=require_notes)
    path = out_dir / "selection.jsonl"
    with path.open("w") as f:
        for a in chosen:
            f.write(json.dumps(a) + "\n")
    (out_dir / "selection_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return path


# --------------------------------------------------------------------------- #
# build: extract selected subjects' rows and write timeline JSONs               #
# --------------------------------------------------------------------------- #

def _clean_value(col: str, v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    if isinstance(v, float) and col in INT_COLUMNS and v.is_integer():
        return int(v)
    if isinstance(v, str):
        if v == "":
            return None
        if col in INT_COLUMNS:
            try:
                return int(float(v))
            except ValueError:
                return v
        if col in FLOAT_COLUMNS:
            try:
                return float(v)
            except ValueError:
                return v
    return v


DATE_ONLY_POLICIES = ("keep", "end_of_day")


def rows_to_event(source_table: str, row: dict, float_value: bool = False,
                  date_only: str = "keep") -> dict:
    """Step 3 + 4: canonical event_time, drop identifiers and nulls.

    date_only: how to treat date-only event times (procedures_icd.chartdate):
      "keep"       -> 'YYYY-MM-DD' as in the CSV (string-sorts before every timed event of
                      that day; same-day-as-admission procedures become pre_admission);
      "end_of_day" -> 'YYYY-MM-DD 23:59:59' (sorts after that day's timed events).
    """
    event_time = None
    for c in EVENT_TIME_PRIORITY:
        v = row.get(c)
        if v not in (None, ""):
            event_time = v
            break
    if date_only == "end_of_day" and isinstance(event_time, str) and len(event_time) == 10:
        event_time = event_time + " 23:59:59"
    ev = {"source_table": source_table}
    if event_time is not None:
        ev["event_time"] = event_time
    for k, v in row.items():
        if k in DROP_COLUMNS or k.startswith("_"):
            continue
        if k == "value" and float_value:
            vv = _clean_value("valuenum", v)
        else:
            vv = _clean_value(k, v)
        if vv is None:
            continue
        ev[k] = vv
    return ev


def _sort_key(item):
    ev, order = item
    et = ev.get("event_time")
    return (et is None, et or "", ev["source_table"], order)


def cmd_build(src: Sources, out_dir: Path, selection: Path, threads: int, mem: str,
              include_subject_admissions: bool = True, tables: list[str] | None = None,
              date_only: str = "keep") -> dict:
    sel = [json.loads(l) for l in selection.read_text().splitlines() if l.strip()]
    subjects = sorted({int(a["subject_id"]) for a in sel})
    hadms = sorted({int(a["hadm_id"]) for a in sel})
    tmp = out_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    ehr_root = out_dir / "ehr_by_subject"
    ehr_root.mkdir(parents=True, exist_ok=True)
    con = _connect(threads, mem, tmp)
    con.execute("CREATE TEMP TABLE sel_subjects(subject_id VARCHAR)")
    con.executemany("INSERT INTO sel_subjects VALUES (?)", [(str(s),) for s in subjects])
    con.execute("CREATE TEMP TABLE sel_hadms(hadm_id VARCHAR)")
    con.executemany("INSERT INTO sel_hadms VALUES (?)", [(str(h),) for h in hadms])

    # (subject, hadm) -> list[(event, order_tuple)]
    events: dict[tuple[str, str], list] = {}
    log = {"tables": {}, "missing": [], "n_selected": len(sel), "n_subjects": len(subjects),
           "date_only": date_only}
    t_all = time.time()
    specs = [TABLE_BY_KEY[k] for k in tables] if tables else TABLES
    for spec in specs:
        p = src.path(spec.root, spec.file)
        if p is None:
            log["missing"].append(spec.file)
            print(f"[build] {spec.key}: MISSING {spec.file}", flush=True)
            continue
        _register_dicts(con, src, _needed_dicts(spec))
        t0 = time.time()
        # Filter by subject (keeps the subject's other admissions for prior-admission history)
        # or by hadm only.
        if include_subject_admissions:
            filt = f"trim({spec.subject_expr}) IN (SELECT subject_id FROM sel_subjects)"
        else:
            filt = f"trim({spec.hadm_expr}) IN (SELECT hadm_id FROM sel_hadms)"
        sel_cols = "src.*"
        extra = f", {spec.extra_select}" if spec.extra_select else ""
        # numeric helper columns for the natural-key tie-break (prefixed "_" -> dropped later)
        okeys = []
        for k in spec.order_keys:
            if k.endswith("_i"):
                base = k[:-2]
                okeys.append(f"TRY_CAST(src.{base} AS BIGINT) AS _o_{base}")
            else:
                okeys.append(f"src.{k} AS _o_{k}")
        okeys_sql = (", " + ", ".join(okeys)) if okeys else ""
        q = (f"SELECT {sel_cols}{extra}{okeys_sql}, trim({spec.subject_expr}) AS _sid, "
             f"trim({spec.hadm_expr}) AS _hid FROM {_from_clause(spec, p)} "
             f"WHERE {_key_filter(spec)} AND {filt}")
        out = tmp / f"extract_{spec.key}.parquet"
        con.execute(f"COPY ({q}) TO '{out}' (FORMAT parquet)")
        df = con.execute(f"SELECT * FROM '{out}'").fetchdf()
        dt = time.time() - t0
        # columns coming from joins may duplicate src ones (emar_detail has no hadm_id; fine)
        cols = list(df.columns)
        okcols = [c for c in cols if c.startswith("_o_")]
        n = 0
        for rec in df.to_dict("records"):
            key = (rec["_sid"], rec["_hid"])
            order = tuple((rec[c] is None or (isinstance(rec[c], float) and rec[c] != rec[c]),
                           rec[c] if isinstance(rec[c], (int, float)) and rec[c] == rec[c]
                           else str(rec[c])) for c in okcols)
            ev = rows_to_event(spec.key, rec, spec.float_value, date_only=date_only)
            events.setdefault(key, []).append((ev, order))
            n += 1
        log["tables"][spec.key] = {"rows": n, "seconds": round(dt, 1)}
        print(f"[build] {spec.key}: {n} rows in {dt:.1f}s", flush=True)
        try:
            out.unlink()
        except OSError:
            pass

    # Step 5: group by admission, sort, write
    written = 0
    for (sid, hid), items in events.items():
        if not any(ev["source_table"] == "hosp_admissions_df" for ev, _ in items):
            continue  # rows whose hadm_id has no admissions row
        items.sort(key=_sort_key)
        d = ehr_root / sid
        d.mkdir(parents=True, exist_ok=True)
        tmp_path = d / f".{hid}.json.tmp"
        tmp_path.write_text(json.dumps([ev for ev, _ in items], ensure_ascii=False, default=str),
                            encoding="utf-8")
        os.replace(tmp_path, d / f"{hid}.json")
        written += 1
    log["admission_files_written"] = written
    log["seconds_total"] = round(time.time() - t_all, 1)
    (out_dir / "build_log.json").write_text(json.dumps(log, indent=2))
    print(f"[build] wrote {written} admission files under {ehr_root} in {log['seconds_total']}s")
    return log


def write_manifest(out_dir: Path, selection: Path) -> Path:
    """clean.py-style manifest (env_ready_admissions.jsonl) for the selected admissions."""
    ehr_root = out_dir / "ehr_by_subject"
    path = out_dir / "env_ready_admissions.jsonl"
    with path.open("w") as f:
        for l in selection.read_text().splitlines():
            if not l.strip():
                continue
            a = json.loads(l)
            p = ehr_root / a["subject_id"] / f"{a['hadm_id']}.json"
            if p.exists():
                f.write(json.dumps({**a, "path": str(p)}) + "\n")
    return path


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["count", "select", "build", "all"])
    ap.add_argument("--mimic-dir", required=False, help="MIMIC-IV root containing hosp/ and icu/")
    ap.add_argument("--note-dir", default=None, help="MIMIC-IV-Note root containing note/")
    ap.add_argument("--out-dir", required=True, help="output root (keep outside git)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--mem", default="64GB")
    ap.add_argument("--no-note-filter", action="store_true",
                    help="drop note-dependent criteria (for the notes-free MIMIC-IV demo only)")
    ap.add_argument("--selection", default=None, help="selection.jsonl for build (default: out-dir)")
    ap.add_argument("--date-only", choices=DATE_ONLY_POLICIES, default="keep",
                    help="event_time for date-only rows (procedures_icd.chartdate); default keeps the "
                         "CSV value as App. B.2 describes")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    src = Sources(Path(args.mimic_dir) if args.mimic_dir else Path("."),
                  Path(args.note_dir) if args.note_dir else None)

    if args.command in ("count", "all"):
        cmd_count(src, out_dir, args.threads, args.mem)
    if args.command in ("select", "all"):
        cmd_select(out_dir, args.n, args.seed, require_notes=not args.no_note_filter)
    if args.command in ("build", "all"):
        selection = Path(args.selection) if args.selection else out_dir / "selection.jsonl"
        cmd_build(src, out_dir, selection, args.threads, args.mem, date_only=args.date_only)
        write_manifest(out_dir, selection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
