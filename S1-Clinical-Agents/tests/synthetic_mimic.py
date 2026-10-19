"""Tiny SYNTHETIC MIMIC-IV-shaped dataset (no real patient data) for tests and dry-runs.

Writes gzip CSVs with MIMIC-IV v3.1 / Note v2.2 column names under
  <root>/mimic/{hosp,icu}/*.csv.gz  and  <root>/note/note/*.csv.gz
Every value is invented. Subjects:
  90000001  admission 29000001 (index case: full tables, all 4 note headings,
            ICD-9 dx with leading zero, ICD-10-PCS procedure, ICU stay)
            + earlier admission 29000000 (prior history, with a note)
  90000002  admission 29000002: note lacks "Brief Hospital Course" -> excluded by note filter
  90000003  admission 29000003: no procedures_icd -> excluded (required table)
plus one labevents row with an empty hadm_id (must be dropped, App. B.2 step 2).
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

NOTE_INDEX = """Name:  ___                     Unit No:   ___

Admission Date:  ___              Discharge Date:   ___

Sex:   M

Service: MEDICINE

Chief Complaint:
Black stools and lightheadedness

History of Present Illness:
Synthetic 67 year old man with atrial fibrillation on warfarin who presents with
two days of melena and dizziness. No hematemesis. Found to have hemoglobin 7.1.

Past Medical History:
Atrial fibrillation, hypertension, type 2 diabetes.

Brief Hospital Course:
Admitted with upper GI bleed. Warfarin was held on admission and pantoprazole 40 mg IV
twice daily was started. He was transfused 2 units of packed red cells. On hospital day 2
he underwent EGD which showed a duodenal ulcer that was clipped. Ceftriaxone was started
for a urinary tract infection after the urine culture grew E. coli. Metoprolol was
switched from tartrate to succinate before discharge. Hemoglobin remained stable.

Discharge Medications:
1. Pantoprazole 40 mg PO BID
2. Metoprolol succinate 50 mg PO daily
3. Cefpodoxime 200 mg PO q12h

Discharge Diagnosis:
Duodenal ulcer with hemorrhage
Acute blood loss anemia
Urinary tract infection

Discharge Instructions:
Synthetic instructions.
"""

NOTE_PRIOR = """Chief Complaint:
Palpitations

History of Present Illness:
Synthetic prior visit for new atrial fibrillation.

Brief Hospital Course:
Rate controlled with metoprolol and started on warfarin.

Discharge Diagnosis:
Atrial fibrillation
"""

NOTE_NO_BHC = """Chief Complaint:
Cough

History of Present Illness:
Synthetic cough.

Discharge Diagnosis:
Pneumonia
"""


def _w(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])


def write_synthetic_mimic(root: Path) -> tuple[Path, Path]:
    m = root / "mimic"
    n = root / "note"
    S1, S2, S3 = 90000001, 90000002, 90000003
    H0, H1, H2, H3 = 29000000, 29000001, 29000002, 29000003

    _w(m / "hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
       [[S1, "M", 65, 2150, "2008 - 2010", None], [S2, "F", 40, 2160, "2011 - 2013", None],
        [S3, "F", 70, 2170, "2014 - 2016", None]])
    _w(m / "hosp/admissions.csv.gz",
       ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id",
        "admission_location", "discharge_location", "insurance", "language", "marital_status", "race",
        "edregtime", "edouttime", "hospital_expire_flag"],
       [[S1, H0, "2150-03-01 10:00:00", "2150-03-03 12:00:00", None, "URGENT", "P1", "EMERGENCY ROOM", "HOME",
         "Medicare", "English", "MARRIED", "WHITE", None, None, 0],
        [S1, H1, "2152-06-10 08:00:00", "2152-06-14 15:00:00", None, "EW EMER.", "P2", "EMERGENCY ROOM", "HOME",
         "Medicare", "English", "MARRIED", "WHITE", "2152-06-10 05:00:00", "2152-06-10 08:30:00", 0],
        [S2, H2, "2160-01-05 09:00:00", "2160-01-07 09:00:00", None, "URGENT", "P3", "CLINIC REFERRAL", "HOME",
         "Other", "English", "SINGLE", "ASIAN", None, None, 0],
        [S3, H3, "2170-02-02 11:00:00", "2170-02-04 11:00:00", None, "URGENT", "P4", "CLINIC REFERRAL", "HOME",
         "Other", "English", "SINGLE", "BLACK", None, None, 0]])
    _w(m / "hosp/d_icd_diagnoses.csv.gz", ["icd_code", "icd_version", "long_title"],
       [["5329", 9, "Duodenal ulcer, unspecified as acute or chronic, without mention of hemorrhage"],
        ["0389", 9, "Unspecified septicemia"], ["2851", 9, "Acute posthemorrhagic anemia"],
        ["5990", 9, "Urinary tract infection, site not specified"], ["42731", 9, "Atrial fibrillation"],
        ["J189", 10, "Pneumonia, unspecified organism"], ["I10", 10, "Essential (primary) hypertension"]])
    _w(m / "hosp/d_icd_procedures.csv.gz", ["icd_code", "icd_version", "long_title"],
       [["0DJ08ZZ", 10, "Inspection of Upper Intestinal Tract, Via Natural or Artificial Opening Endoscopic"],
        ["9904", 9, "Transfusion of packed cells"], ["8938", 9, "Other nonoperative respiratory measurements"]])
    _w(m / "hosp/diagnoses_icd.csv.gz", ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
       [[S1, H1, 1, "5329", 9], [S1, H1, 2, "2851", 9], [S1, H1, 3, "5990", 9], [S1, H1, 4, "42731", 9],
        [S1, H1, 5, "0389", 9], [S1, H0, 1, "42731", 9], [S2, H2, 1, "J189", 10], [S3, H3, 1, "I10", 10]])
    _w(m / "hosp/procedures_icd.csv.gz", ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
       [[S1, H1, 1, "2152-06-11", "0DJ08ZZ", 10], [S1, H1, 2, "2152-06-10", "9904", 9],
        [S2, H2, 1, "2160-01-05", "8938", 9]])
    _w(m / "hosp/d_labitems.csv.gz", ["itemid", "label", "fluid", "category"],
       [[51222, "Hemoglobin", "Blood", "Hematology"], [50983, "Sodium", "Blood", "Chemistry"],
        [51237, "INR(PT)", "Blood", "Hematology"], [50912, "Creatinine", "Blood", "Chemistry"]])
    lab_h = ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime",
             "storetime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag",
             "priority", "comments"]
    lab = [
        [1, S1, H1, 11, 51222, None, "2152-06-10 06:00:00", "2152-06-10 06:30:00", "7.1", 7.1, "g/dL", 13.7, 17.5, "abnormal", "STAT", None],
        [2, S1, H1, 11, 51237, None, "2152-06-10 06:00:00", "2152-06-10 06:30:00", "3.4", 3.4, None, 0.9, 1.1, "abnormal", "STAT", None],
        [3, S1, H1, 11, 50983, None, "2152-06-10 06:00:00", "2152-06-10 06:30:00", "138", 138, "mEq/L", 133, 145, None, "STAT", None],
        [4, S1, H1, 12, 51222, None, "2152-06-10 20:00:00", "2152-06-10 20:30:00", "8.4", 8.4, "g/dL", 13.7, 17.5, "abnormal", "ROUTINE", None],
        [5, S1, H1, 13, 51222, None, "2152-06-11 06:00:00", "2152-06-11 06:30:00", "8.6", 8.6, "g/dL", 13.7, 17.5, "abnormal", "ROUTINE", None],
        [6, S1, H1, 13, 50912, None, "2152-06-11 06:00:00", "2152-06-11 06:30:00", "1.0", 1.0, "mg/dL", 0.5, 1.2, None, "ROUTINE", None],
        [7, S1, H1, 14, 51222, None, "2152-06-13 06:00:00", "2152-06-13 06:30:00", "9.0", 9.0, "g/dL", 13.7, 17.5, "abnormal", "ROUTINE", None],
        [8, S1, None, 15, 51222, None, "2152-07-01 06:00:00", "2152-07-01 06:30:00", "12.0", 12.0, "g/dL", 13.7, 17.5, None, "ROUTINE", None],
        [9, S1, H0, 16, 50983, None, "2150-03-01 11:00:00", "2150-03-01 11:30:00", "140", 140, "mEq/L", 133, 145, None, "STAT", None],
        [10, S2, H2, 17, 51222, None, "2160-01-05 10:00:00", "2160-01-05 10:30:00", "12.5", 12.5, "g/dL", 11.0, 15.0, None, "STAT", None],
        [11, S3, H3, 18, 50983, None, "2170-02-02 12:00:00", "2170-02-02 12:30:00", "141", 141, "mEq/L", 133, 145, None, "STAT", None],
    ]
    _w(m / "hosp/labevents.csv.gz", lab_h, lab)
    _w(m / "hosp/microbiologyevents.csv.gz",
       ["microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "order_provider_id", "chartdate", "charttime",
        "spec_itemid", "spec_type_desc", "test_seq", "storedate", "storetime", "test_itemid", "test_name",
        "org_itemid", "org_name", "isolate_num", "quantity", "ab_itemid", "ab_name", "dilution_text",
        "dilution_comparison", "dilution_value", "interpretation", "comments"],
       [[1, S1, H1, 21, None, "2152-06-11 00:00:00", "2152-06-11 09:00:00", 70079, "URINE", 1,
         "2152-06-12 00:00:00", "2152-06-12 10:00:00", 90039, "URINE CULTURE", 80002, "ESCHERICHIA COLI", 1,
         None, 90, "CEFTRIAXONE", "<=1", "<=", 1, "S", None]])
    rx_h = ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq", "order_provider_id", "starttime", "stoptime",
            "drug_type", "drug", "formulary_drug_cd", "gsn", "ndc", "prod_strength", "form_rx", "dose_val_rx",
            "dose_unit_rx", "form_val_disp", "form_unit_disp", "doses_per_24_hrs", "route"]
    _w(m / "hosp/prescriptions.csv.gz", rx_h, [
        [S1, H1, 101, "X-1", 1, None, "2152-06-10 09:00:00", "2152-06-14 12:00:00", "MAIN", "Pantoprazole", "PANT40I", None, None, "40 mg Vial", None, "40", "mg", "1", "VIAL", 2, "IV"],
        [S1, H1, 102, "X-2", 2, None, "2152-06-10 10:00:00", "2152-06-10 16:00:00", "MAIN", "Packed Red Blood Cells", None, None, None, "1 unit", None, "1", "unit", "1", "UNIT", 1, "IV"],
        [S1, H1, 103, "X-3", 3, None, "2152-06-12 11:00:00", "2152-06-14 12:00:00", "MAIN", "CeftriaXONE", "CEFT1I", None, None, "1 g Vial", None, "1", "g", "1", "VIAL", 1, "IV"],
        [S1, H1, 104, "X-4", 4, None, "2152-06-10 09:00:00", "2152-06-13 09:00:00", "MAIN", "Metoprolol Tartrate", "METO25", None, None, "25 mg Tablet", None, "25", "mg", "1", "TAB", 2, "PO"],
        [S1, H1, 105, "X-5", 5, None, "2152-06-13 09:00:00", "2152-06-14 12:00:00", "MAIN", "Metoprolol Succinate XL", "METS50", None, None, "50 mg Tablet", None, "50", "mg", "1", "TAB", 1, "PO"],
        [S1, H0, 106, "X-6", 1, None, "2150-03-01 12:00:00", "2150-03-03 12:00:00", "MAIN", "Warfarin", "WARF5", None, None, "5 mg Tablet", None, "5", "mg", "1", "TAB", 1, "PO"],
        [S2, H2, 108, "X-8", 1, None, "2160-01-05 12:00:00", "2160-01-07 09:00:00", "MAIN", "Azithromycin", "AZIT250", None, None, "250 mg Tablet", None, "500", "mg", "2", "TAB", 1, "PO"],
        [S3, H3, 107, "X-7", 1, None, "2170-02-02 12:00:00", "2170-02-04 11:00:00", "MAIN", "Lisinopril", "LISI10", None, None, "10 mg Tablet", None, "10", "mg", "1", "TAB", 1, "PO"],
    ])
    _w(m / "hosp/pharmacy.csv.gz",
       ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "medication", "proc_type",
        "status", "entertime", "verifiedtime", "route", "frequency", "disp_sched", "infusion_type",
        "sliding_scale", "lockout_interval", "basal_rate", "one_hr_max", "doses_per_24_hrs", "duration",
        "duration_interval", "expiration_value", "expiration_unit", "expirationdate", "dispensation", "fill_quantity"],
       [[S1, H1, 101, "X-1", "2152-06-10 09:00:00", "2152-06-14 12:00:00", "Pantoprazole", "Unit Dose", "Discontinued via patient discharge", "2152-06-10 08:50:00", "2152-06-10 08:55:00", "IV", "Q12H", None, None, None, None, None, None, 2, None, "Ongoing", None, None, None, None, None],
        [S1, H1, 103, "X-3", "2152-06-12 11:00:00", "2152-06-14 12:00:00", "CeftriaXONE", "IV Piggyback", "Discontinued via patient discharge", "2152-06-12 10:50:00", "2152-06-12 10:55:00", "IV", "Q24H", None, None, None, None, None, None, 1, None, "Ongoing", None, None, None, None, None],
        [S1, H1, 104, "X-4", "2152-06-10 09:00:00", "2152-06-13 09:00:00", "Metoprolol Tartrate", "Unit Dose", "Discontinued", "2152-06-10 08:50:00", "2152-06-10 08:55:00", "PO", "BID", None, None, None, None, None, None, 2, None, "Ongoing", None, None, None, None, None]])
    _w(m / "hosp/emar.csv.gz",
       ["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id", "enter_provider_id", "charttime",
        "medication", "event_txt", "scheduletime", "storetime"],
       [[S1, H1, f"{S1}-1", 1, "X-1", 101, None, "2152-06-10 09:30:00", "Pantoprazole", "Administered", "2152-06-10 09:00:00", "2152-06-10 09:31:00"],
        [S1, H1, f"{S1}-2", 2, "X-3", 103, None, "2152-06-12 11:30:00", "CeftriaXONE", "Administered", "2152-06-12 11:00:00", "2152-06-12 11:31:00"],
        [S1, H1, f"{S1}-3", 3, "X-4", 104, None, "2152-06-10 09:40:00", "Metoprolol Tartrate", "Held", "2152-06-10 09:00:00", "2152-06-10 09:41:00"]])
    _w(m / "hosp/emar_detail.csv.gz",
       ["subject_id", "emar_id", "emar_seq", "parent_field_ordinal", "administration_type", "pharmacy_id",
        "barcode_type", "reason_for_no_barcode", "complete_dose_not_given", "dose_due", "dose_due_unit",
        "dose_given", "dose_given_unit", "will_remainder_of_dose_be_given", "product_amount_given",
        "product_unit", "product_code", "product_description", "product_description_other",
        "prior_infusion_rate", "infusion_rate", "infusion_rate_adjustment", "infusion_rate_adjustment_amount",
        "infusion_rate_unit", "route", "infusion_complete", "completion_interval", "new_iv_bag_hung",
        "continued_infusion_in_other_location", "restart_interval", "side", "site", "non_formulary_visual_verification"],
       [[S1, f"{S1}-1", 1, None, "IV Bolus", 101, "FP", None, None, "40", "mg", None, None, None, None, None, None, None, None, None, None, None, None, None, "IV", None, None, None, None, None, None, None, None],
        [S1, f"{S1}-1", 1, "1.1", "IV Bolus", 101, "FP", None, None, None, None, "40", "mg", None, "1", "VIAL", "PANT40I", "Pantoprazole 40mg Vial", None, None, None, None, None, None, "IV", None, None, None, None, None, None, None, None],
        [S1, f"{S1}-2", 2, None, "IV Piggyback", 103, "FP", None, None, "1", "g", "1", "g", None, None, None, None, None, None, None, None, None, None, None, "IV", None, None, None, None, None, None, None, None]])
    _w(m / "icu/d_items.csv.gz",
       ["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"],
       [[220045, "Heart Rate", "HR", "chartevents", "Routine Vital Signs", "bpm", "Numeric", None, None],
        [220179, "Non Invasive Blood Pressure systolic", "NBPs", "chartevents", "Routine Vital Signs", "mmHg", "Numeric", None, None],
        [225158, "NaCl 0.9%", "NaCl 0.9%", "inputevents", "Fluids/Intake", "mL", "Solution", None, None],
        [226559, "Foley", "Foley", "outputevents", "Output", "mL", "Numeric", None, None],
        [224277, "18 Gauge", "18 Gauge", "procedureevents", "Access Lines - Peripheral", None, "Process", None, None],
        [225754, "18 Gauge Insertion Date", "18 Gauge Insertion Date", "datetimeevents", "Access Lines - Peripheral", None, "Date time", None, None],
        [220490, "Water", "Water", "ingredientevents", "Ingredients", "mL", "Ingredient", None, None]])
    _w(m / "icu/chartevents.csv.gz",
       ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"],
       [[S1, H1, 39000001, 1, "2152-06-10 08:15:00", "2152-06-10 08:20:00", 220045, "112", 112, "bpm", 0],
        [S1, H1, 39000001, 1, "2152-06-10 08:15:00", "2152-06-10 08:20:00", 220179, "92", 92, "mmHg", 0],
        [S1, H1, 39000001, 1, "2152-06-10 12:00:00", "2152-06-10 12:05:00", 220045, "98", 98, "bpm", 0]])
    _w(m / "icu/datetimeevents.csv.gz",
       ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valueuom", "warning"],
       [[S1, H1, 39000001, 1, "2152-06-10 08:30:00", "2152-06-10 08:31:00", 225754, "2152-06-10 08:00:00", "Date", 0]])
    io_h = ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime", "storetime", "itemid",
            "amount", "amountuom", "rate", "rateuom", "orderid", "linkorderid", "statusdescription",
            "originalamount", "originalrate"]
    _w(m / "icu/inputevents.csv.gz", io_h,
       [[S1, H1, 39000001, 1, "2152-06-10 08:40:00", "2152-06-10 09:40:00", "2152-06-10 08:41:00", 225158, 1000, "ml", 1000, "mL/hour", 5001, 5001, "FinishedRunning", 1000, 1000]])
    _w(m / "icu/ingredientevents.csv.gz", io_h,
       [[S1, H1, 39000001, 1, "2152-06-10 08:40:00", "2152-06-10 09:40:00", "2152-06-10 08:41:00", 220490, 1000, "ml", 1000, "mL/hour", 5001, 5001, "FinishedRunning", 1000, 1000]])
    _w(m / "icu/outputevents.csv.gz",
       ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valueuom"],
       [[S1, H1, 39000001, 1, "2152-06-10 14:00:00", "2152-06-10 14:01:00", 226559, 350, "ml"]])
    _w(m / "icu/procedureevents.csv.gz",
       ["subject_id", "hadm_id", "stay_id", "caregiver_id", "starttime", "endtime", "storetime", "itemid", "value",
        "valueuom", "location", "locationcategory", "orderid", "linkorderid", "ordercategoryname",
        "ordercategorydescription", "patientweight", "isopenbag", "continueinnextdept", "statusdescription",
        "ORIGINALAMOUNT", "ORIGINALRATE"],
       [[S1, H1, 39000001, 1, "2152-06-10 08:00:00", "2152-06-12 08:00:00", "2152-06-12 08:01:00", 224277, 2880, "min", "L Antecube", "Upper", 6001, 6001, "Peripheral Lines", "Task", 80, 0, 0, "FinishedRunning", 0, 0]])

    note_h = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"]
    _w(n / "note/discharge.csv.gz", note_h,
       [[f"{S1}-DS-1", S1, H0, "DS", 1, "2150-03-03 00:00:00", "2150-03-03 15:00:00", NOTE_PRIOR],
        [f"{S1}-DS-2", S1, H1, "DS", 2, "2152-06-14 00:00:00", "2152-06-14 18:00:00", NOTE_INDEX],
        [f"{S2}-DS-1", S2, H2, "DS", 1, "2160-01-07 00:00:00", "2160-01-07 15:00:00", NOTE_NO_BHC],
        [f"{S3}-DS-1", S3, H3, "DS", 1, "2170-02-04 00:00:00", "2170-02-04 15:00:00", NOTE_PRIOR]])
    _w(n / "note/radiology.csv.gz", note_h,
       [[f"{S1}-RR-1", S1, H1, "RR", 1, "2152-06-10 07:00:00", "2152-06-10 08:00:00",
         "EXAMINATION: CHEST (PORTABLE AP)\nIMPRESSION: No acute cardiopulmonary process."]])
    return m, n


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1])
    print(write_synthetic_mimic(out))
