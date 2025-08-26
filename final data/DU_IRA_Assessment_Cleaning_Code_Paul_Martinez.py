"""
University of Denver – Office of Institutional Research & Analysis
Data Scientist Technical Assessment

Author: Paul Martinez
Environment: Python 3.13.x (IDLE, Windows)
Date: 2025-08-12

Purpose:
    End-to-end, reproducible pipeline to prepare a student-level dataset
    from enrollment, grades, and program mapping files, and a final CSV export.
    Step-by-step reproduction instructions are included for future analysts.

Notes:
    - Dependencies: pandas, numpy, statsmodels (matplotlib if figures)
    - Modular design for re-running with new term data
    - Statistical test: two-proportion z-test via statsmodels
    - Output: student_level_final_data.csv with aggregated fields
"""

# -----------------------------
# Step 1 — Install Required Packages (PowerShell, Python 3.13)
# -----------------------------
# py -3.13 -m pip install pandas numpy 

# -----------------------------
# Imports
# -----------------------------
import pandas as pd
import numpy as np
import os

# -----------------------------
# Step 2 — Load Input Files
# -----------------------------
# Assuming file names never change; only update the folder path if needed.
DATA_FOLDER = r"C:\Users\paul6\OneDrive\Desktop\original data"

# File paths
fall_enrollment_path = os.path.join(DATA_FOLDER, "fall_enrollment.csv")
grades_path          = os.path.join(DATA_FOLDER, "grades.csv")
program_data_path    = os.path.join(DATA_FOLDER, "program_data.csv")

# Presence check
for file_path in [fall_enrollment_path, grades_path, program_data_path]:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
    else:
        print(f"[OK] Found: {file_path}")

# Load
fall_enrollment = pd.read_csv(fall_enrollment_path)
grades          = pd.read_csv(grades_path)
program_data    = pd.read_csv(program_data_path)

print(
    "Loaded shapes -> "
    f"fall_enrollment: {fall_enrollment.shape}, "
    f"grades: {grades.shape}, "
    f"program_data: {program_data.shape}"
)

# -----------------------------
# Step 3 — Standardize keys, check coverage, and merge by (id, term_code)
# -----------------------------
# Normalize join keys (string + trim)
for df, name in [(fall_enrollment, "fall_enrollment"), (grades, "grades")]:
    for col in ["id", "term_code"]:
        if col not in df.columns:
            raise KeyError(f"[FATAL] '{col}' not found in {name}")
        df[col] = df[col].astype(str).str.strip()

# Visibility on terms present
print("[INFO] Terms in enrollment:", sorted(fall_enrollment["term_code"].unique())[:10])
print("[INFO] Terms in grades:    ", sorted(grades["term_code"].unique())[:10])

# Coverage check (id, term_code)
enroll_keys = set(map(tuple, fall_enrollment[["id", "term_code"]].values))
grade_keys  = set(map(tuple, grades[["id", "term_code"]].values))
matched     = enroll_keys & grade_keys

print("[COVERAGE] (id, term_code) match results")
print(f"  enrollment keys: {len(enroll_keys)}")
print(f"  grades keys:     {len(grade_keys)}")
print(f"  matched:         {len(matched)} ({len(matched)/max(len(enroll_keys),1):.1%} of enrollment)")

miss_enroll_has_no_grades = list(enroll_keys - grade_keys)[:10]
miss_grades_no_enroll     = list(grade_keys - enroll_keys)[:10]
if miss_enroll_has_no_grades:
    print("  sample enrollment->no grades (up to 10):", miss_enroll_has_no_grades)
if miss_grades_no_enroll:
    print("  sample grades->no enrollment (up to 10):", miss_grades_no_enroll)

# Aggregate grades to term (avoid row multiplication; GPA later)
grades_term = (
    grades.groupby(["id", "term_code"], as_index=False)
          .agg(course_count=("id", "size"))
)

# Left-join to keep all enrollment records
enroll_plus_gr = fall_enrollment.merge(
    grades_term,
    on=["id", "term_code"],
    how="left",
    validate="m:1"
)

print("[MERGE] enrollment ← grades_term (by id, term_code)")
print("  rows:", enroll_plus_gr.shape[0])
print("  unique (id, term):", enroll_plus_gr[["id","term_code"]].drop_duplicates().shape[0])
print("  course_count nulls:", enroll_plus_gr["course_count"].isna().sum(), "(no grades for that term)")

# -----------------------------
# Step 4 — Attach Program (program_data) via (college, degree, major)
# -----------------------------
# Standardize text keys (trim + uppercase) on both sides to ensure a reliable match
enroll_plus_gr["college_key"] = enroll_plus_gr["college"].astype(str).str.strip().str.upper()
enroll_plus_gr["degree_key"]  = enroll_plus_gr["degr"].astype(str).str.strip().str.upper()
enroll_plus_gr["major_key"]   = enroll_plus_gr["majr"].astype(str).str.strip().str.upper()

program_data["college_key"] = program_data["COLLEGE"].astype(str).str.strip().str.upper()
program_data["degree_key"]  = program_data["DEGREE"].astype(str).str.strip().str.upper()
program_data["major_key"]   = program_data["MAJOR"].astype(str).str.strip().str.upper()

# Left join to add PROGRAM
df_final = enroll_plus_gr.merge(
    program_data[["college_key","degree_key","major_key","PROGRAM"]],
    on=["college_key","degree_key","major_key"],
    how="left",
    validate="m:1"
)

matched_programs = df_final["PROGRAM"].notna().sum()
total_rows = len(df_final)
print(f"[MERGE] PROGRAM matched for {matched_programs} of {total_rows} records.")
print(f"[MERGE] PROGRAM missing for {total_rows - matched_programs} records.")
# -----------------------------
# Step 5 — Term GPA, census flags, DU-defined demographics, and age
# -----------------------------

# 5a) Term GPA from letter grades
GRADE_POINTS = {
    "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "F": 0.0
}
grades_gpa = grades.copy()
grades_gpa["final_course_grade"] = grades_gpa["final_course_grade"].astype(str).str.strip()
grades_gpa["grade_points"] = grades_gpa["final_course_grade"].map(GRADE_POINTS)
grades_gpa = grades_gpa.dropna(subset=["grade_points"])

term_gpa = (
    grades_gpa.groupby(["id", "term_code"], as_index=False)
              .agg(term_gpa=("grade_points", "mean"),
                   course_count=("grade_points", "size"))
)
print("[INFO] term_gpa rows:", term_gpa.shape[0])

# 5b) Split census exactly as provided: WK3 vs EOT
w3 = (df_final[df_final["census"] == "WK3"]
      .sort_values(["id", "term_code"])
      .drop_duplicates(["id", "term_code"])[
          ["id","term_code","majr","degr","college","PROGRAM",
           "race_desc","ethn_desc","visa_desc","legal_sex_desc","birth_date"]
      ].rename(columns={
          "majr":"majr_w3","degr":"degr_w3","college":"college_w3","PROGRAM":"PROGRAM_w3",
          "race_desc":"race_desc_w3","ethn_desc":"ethn_desc_w3","visa_desc":"visa_desc_w3",
          "legal_sex_desc":"legal_sex_desc_w3","birth_date":"birth_date_w3"
      }))

end = (df_final[df_final["census"] == "EOT"]
       .sort_values(["id", "term_code"])
       .drop_duplicates(["id", "term_code"])[
           ["id","term_code","majr","degr","college","PROGRAM",
            "race_desc","ethn_desc","visa_desc","legal_sex_desc","birth_date"]
       ].rename(columns={
           "majr":"majr_end","degr":"degr_end","college":"college_end","PROGRAM":"PROGRAM_end",
           "race_desc":"race_desc_end","ethn_desc":"ethn_desc_end","visa_desc":"visa_desc_end",
           "legal_sex_desc":"legal_sex_desc_end","birth_date":"birth_date_end"
       }))

# 5c) Merge WK3 and EOT; create flags
student_term = pd.merge(w3, end, on=["id", "term_code"], how="outer")

student_term["enrolled_w3"]  = student_term[["majr_w3","degr_w3","college_w3"]].notna().any(axis=1)
student_term["enrolled_end"] = student_term[["majr_end","degr_end","college_end"]].notna().any(axis=1)
student_term["persisted_w3_to_end"] = student_term["enrolled_w3"] & student_term["enrolled_end"]

# UN → non-UN only for those who persisted
student_term["undeclared_to_declared"] = (
    student_term["persisted_w3_to_end"] &
    student_term["majr_w3"].fillna("").str.upper().str.startswith("UN") &
    ~student_term["majr_end"].fillna("").str.upper().str.startswith("UN")
)

# Prefer EOT info if they persisted; otherwise fall back to WK3
student_term["PROGRAM"] = np.where(student_term["enrolled_end"], student_term["PROGRAM_end"], student_term["PROGRAM_w3"])
student_term["degree"]  = np.where(student_term["enrolled_end"], student_term["degr_end"],    student_term["degr_w3"])
student_term["major"]   = np.where(student_term["enrolled_end"], student_term["majr_end"],    student_term["majr_w3"])
student_term["college"] = np.where(student_term["enrolled_end"], student_term["college_end"], student_term["college_w3"])
student_term["gender"]  = np.where(student_term["legal_sex_desc_end"].notna(),
                                   student_term["legal_sex_desc_end"], student_term["legal_sex_desc_w3"])

# 5f) Race/ethnicity (visa → ethnicity → race). Missing/blank visa ≠ International.
NON_INTL_VISA_TYPES = {"PR", "RF", "AS"}  # only present AND not in this set → International

def _race_ethnicity_du(row) -> str:
    # Prefer WK3 values; fallback to EOT
    visa_raw = row.get("visa_desc_w3") if pd.notna(row.get("visa_desc_w3")) else row.get("visa_desc_end")
    ethn_raw = row.get("ethn_desc_w3") if pd.notna(row.get("ethn_desc_w3")) else row.get("ethn_desc_end")
    race_raw = row.get("race_desc_w3") if pd.notna(row.get("race_desc_w3")) else row.get("race_desc_end")

    # Clean only if non-missing and non-blank; otherwise set ""
    visa = (str(visa_raw).strip().upper() if (pd.notna(visa_raw) and str(visa_raw).strip() != "") else "")
    ethn = (str(ethn_raw).strip()            if (pd.notna(ethn_raw) and str(ethn_raw).strip() != "") else "")
    race = (str(race_raw).strip()            if (pd.notna(race_raw) and str(race_raw).strip() != "") else "")

    # 1) Visa precedence
    if visa and visa not in NON_INTL_VISA_TYPES:
        return "International"   # e.g., B1, J1, R1

    # 2) Ethnicity
    if ethn.lower() == "hispanic or latino":
        return "Hispanic or Latino"

    # 3) Otherwise race (or Unknown)
    return race if race else "Unknown"

student_term["race_ethnicity"] = student_term.apply(_race_ethnicity_du, axis=1)


# 5e) Age — month-level anchor 
# Map DU term-code suffixes to an "as-of" month; use day=1 to avoid overprecision.
_TERM_ASOF_MONTH = {10: 10, 70: 10, 20: 3, 30: 7, 40: 7, 50: 7}  # Fall=Oct, Spring=Mar, Summer=Jul

def _asof_from_term(term_code):
    s = str(term_code)
    if len(s) >= 6 and s[:4].isdigit():
        year = int(s[:4])
        try:
            suff = int(s[-2:])
        except ValueError:
            return pd.NaT
        month = _TERM_ASOF_MONTH.get(suff, 10)  # default to Oct if unknown
        return pd.Timestamp(year=year, month=month, day=1)
    return pd.NaT

def _age_years(dob_w3, dob_end, term_code):
    dob = dob_w3 if pd.notna(dob_w3) else dob_end
    dt = pd.to_datetime(dob, errors="coerce")
    asof = _asof_from_term(term_code)
    if pd.isna(dt) or pd.isna(asof):
        return np.nan
    # integer age at as-of date (month-level anchor)
    years = asof.year - dt.year
    if (asof.month, asof.day) < (dt.month, dt.day):
        years -= 1
    return years

student_term["age"] = student_term.apply(
    lambda r: _age_years(r.get("birth_date_w3"), r.get("birth_date_end"), r.get("term_code")),
    axis=1
)


# 5f) Attach term GPA
student_term = student_term.merge(term_gpa, on=["id","term_code"], how="left", validate="m:1")

print("[BUILD] student_term:", student_term.shape)
print("[BUILD] persisted_w3_to_end =", int(student_term["persisted_w3_to_end"].sum()))
print("[BUILD] undeclared_to_declared =", int(student_term["undeclared_to_declared"].sum()))
print("[BUILD] term_gpa non-missing =", student_term["term_gpa"].notna().sum())




# -----------------------------
# Step 6 — Collapse to one row per (id, term_code)
# -----------------------------
# Keep only the fields needed to answer the deliverables

student_level = student_term[[
    "id", "term_code",
    "gender", "race_ethnicity", "age",
    "college", "degree", "major", "PROGRAM",
    "term_gpa", "course_count",
    "persisted_w3_to_end", "undeclared_to_declared"
]].copy()

# Round GPA to 2 decimals (values only)
student_level["term_gpa"] = student_level["term_gpa"].round(2)

# Rename the column header only (do NOT change values)
student_level = student_level.rename(columns={"PROGRAM": "program"})

# Ensure one row per student-term
dups = student_level.duplicated(subset=["id", "term_code"], keep=False)
if dups.any():
    dup_count = int(dups.sum())
    print(f"[STUDENT_LEVEL] duplicates at (id, term_code): {dup_count} — keeping first occurrence")
    student_level = student_level.drop_duplicates(subset=["id", "term_code"], keep="first")

print("[STUDENT_LEVEL] rows:", student_level.shape[0])
print("[STUDENT_LEVEL] unique (id, term):", student_level[["id", "term_code"]].drop_duplicates().shape[0])


# -----------------------------
# Step 7 — Export student-level dataset to CSV
# -----------------------------
# NOTE: Change OUTPUT_FOLDER to your preferred location

# Uses the 'os' import from earlier in the script
OUTPUT_FOLDER = r"C:\Users\paul6\OneDrive\Desktop\final data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

out_path = os.path.join(OUTPUT_FOLDER, "student_level_final_data.csv")
student_level.to_csv(out_path, index=False)
print(f"[EXPORT] student_level_final_data.csv written to: {out_path}")



