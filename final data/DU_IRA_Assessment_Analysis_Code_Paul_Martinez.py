"""
University of Denver – Office of Institutional Research & Analysis
Data Scientist Technical Assessment

Author: Paul Martinez
Environment: Python 3.13.x (IDLE, Windows)
Date: 2025-08-12

Purpose:
    This script performs the analysis and reporting phase of the technical assessment
    using the cleaned and aggregated student-level dataset prepared earlier.
    All calculations, statistical tests, and visualizations are generated from
    the final dataset to address the deliverable questions. Note this code is
    designed to be fully automated by creating a csv export from all analysis.
    More specifically, a table/graph per sheet. 

Notes:
    - Dependencies: pandas, numpy, statsmodels, matplotlib, seaborn (optional for visuals)
    - Input: student_level_final_data.csv from the "final data" folder
    - Output: statistical results, tables, and figures
    - This script assumes the cleaning pipeline has already been run and exported the final dataset.
"""

# -------- Setup --------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

FINAL_DATA_FOLDER = r"C:\Users\paul6\OneDrive\Desktop\final data"
in_path  = os.path.join(FINAL_DATA_FOLDER, "student_level_final_data.csv")
out_xlsx = os.path.join(FINAL_DATA_FOLDER, "DU_IRA_Assessment_Report.xlsx")
os.makedirs(FINAL_DATA_FOLDER, exist_ok=True)

df = pd.read_csv(in_path)

# -------- Helper: write sheet + (optional) embed image --------
def write_sheet(writer, sheet_name, table_df, image_path=None, image_cell="G2", img_scale=1.0):
    table_df.to_excel(writer, sheet_name=sheet_name, index=False)
    if image_path is not None and os.path.exists(image_path):
        ws = writer.sheets[sheet_name]
        ws.insert_image(image_cell, image_path, {"x_scale": img_scale, "y_scale": img_scale})

# ==============================
# 1) Persistence rate (table)
# ==============================
total_students = len(df)
persisted = int(df["persisted_w3_to_end"].sum())
persistence_rate = persisted / total_students if total_students else np.nan

tbl_persistence = pd.DataFrame({
    "metric": ["total_students", "persisted_students", "persistence_rate"],
    "value":  [total_students, persisted, round(persistence_rate, 6)]
})

# ==============================================
# 2) Gender difference in persistence (tables)
# ==============================================
by_gender = (df.groupby("gender")["persisted_w3_to_end"]
               .agg(n_total="count", n_persisted="sum"))
by_gender["persist_rate"] = (by_gender["n_persisted"] / by_gender["n_total"]).round(6)
tbl_gender = by_gender.reset_index()

# Two-proportion z-test for Male vs Female (if both present)
ztest_table = pd.DataFrame(columns=["comparison", "z_stat", "p_value"])
if {"Male","Female"}.issubset(set(tbl_gender["gender"])):
    male = tbl_gender.loc[tbl_gender["gender"]=="Male", ["n_persisted","n_total"]].iloc[0]
    female = tbl_gender.loc[tbl_gender["gender"]=="Female", ["n_persisted","n_total"]].iloc[0]
    z_stat, p_val = proportions_ztest(count=[male["n_persisted"], female["n_persisted"]],
                                      nobs=[male["n_total"], female["n_total"]])
    ztest_table = pd.DataFrame([{
        "comparison": "Male vs Female",
        "z_stat": round(float(z_stat), 6),
        "p_value": round(float(p_val), 6)
    }])
else:
    ztest_table = pd.DataFrame([{"comparison": "Male vs Female", "z_stat": np.nan, "p_value": np.nan}])

# =====================================================
# 3) Class makeup: race/ethnicity and gender (tables)
# =====================================================
race_counts = (df["race_ethnicity"].value_counts(dropna=False)
               .rename_axis("race_ethnicity").reset_index(name="count"))
race_counts["percent"] = (race_counts["count"] / total_students * 100).round(2)

gender_counts = (df["gender"].value_counts(dropna=False)
                 .rename_axis("gender").reset_index(name="count"))
gender_counts["percent"] = (gender_counts["count"] / total_students * 100).round(2)

race_gender_counts = pd.crosstab(df["race_ethnicity"], df["gender"]).reset_index()
race_gender_pct = (pd.crosstab(df["race_ethnicity"], df["gender"], normalize="index")*100
                   ).round(2).reset_index()

# =================================
# 4) Age distribution (table + fig)
# =================================
age_summary = df["age"].describe().round(2).rename_axis("stat").reset_index(name="value")

# Histogram
age_plot = os.path.join(FINAL_DATA_FOLDER, "age_distribution.png")
plt.figure(figsize=(8,5))
plt.hist(df["age"].dropna(), bins=10, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(age_plot, dpi=150)
plt.close()

# ===============================================
# 5) Term GPA distribution (table + optional fig)
# ===============================================
gpa_summary = df["term_gpa"].describe().round(3).rename_axis("stat").reset_index(name="value")
gpa_counts = (df["term_gpa"].value_counts().sort_index()
              .rename_axis("term_gpa").reset_index(name="count"))
gpa_counts["percent"] = (gpa_counts["count"] / total_students * 100).round(2)

# Histogram
gpa_plot = os.path.join(FINAL_DATA_FOLDER, "gpa_distribution.png")
plt.figure(figsize=(8,5))
plt.hist(df["term_gpa"].dropna(), bins=10, edgecolor="black")
plt.title("Term GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Number of Students")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(gpa_plot, dpi=150)
plt.close()

# ======================================
# 6) Average GPA by program (table only)
# ======================================
gpa_by_program = (df.groupby("program", dropna=False)["term_gpa"].mean()
                    .round(2).reset_index().rename(columns={"term_gpa":"average_gpa"}))

# =========================================================
# 7) Visual: course grade distributions by degree (fig+tbl)
# =========================================================
target_degrees = ["BA", "BS", "BM", "BFA"]
df_deg = df[df["degree"].isin(target_degrees)].copy()

# Stacked histo via matplotlib (simple + reliable)
degree_plot = os.path.join(FINAL_DATA_FOLDER, "degree_grade_distributions.png")
plt.figure(figsize=(10,6))
bins = np.linspace(0, 4.0, 21)  # 0.2 GPA bins
for deg in target_degrees:
    plt.hist(df_deg.loc[df_deg["degree"]==deg, "term_gpa"].dropna(),
             bins=bins, alpha=0.7, label=deg, edgecolor="black")
plt.title("Course Grade Distribution by Broad Degree Level")
plt.xlabel("Term GPA")
plt.ylabel("Number of Students")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Degree", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(degree_plot, dpi=150)
plt.close()

# Underlying counts table (so the visual has tabular backing)
bin_labels = [f"[{round(bins[i],1)}, {round(bins[i+1],1)})" for i in range(len(bins)-1)]
df_deg["gpa_bin"] = pd.cut(df_deg["term_gpa"], bins=bins, labels=bin_labels, include_lowest=True, right=False)
degree_bin_counts = (df_deg.groupby(["degree","gpa_bin"]).size()
                     .reset_index(name="count"))

# ====================================================================
# 8) Proportion: WK3 undeclared → declared by EOT (table, no graphic)
# ====================================================================
# Final dataset includes boolean 'undeclared_to_declared'.
n_undecl_to_decl = int(df["undeclared_to_declared"].sum())
persisters = int(df["persisted_w3_to_end"].sum())

# Report practical rates we can compute directly from the final dataset
prop_of_total = n_undecl_to_decl / total_students if total_students else np.nan
prop_of_persisters = n_undecl_to_decl / persisters if persisters else np.nan

undecl_tbl = pd.DataFrame({
    "metric": ["n_undecl_to_decl", "prop_of_total", "prop_of_persisters",
               "note"],
    "value":  [n_undecl_to_decl,
               round(prop_of_total, 6),
               round(prop_of_persisters, 6),
               "Denominator of *all WK3-undeclared* not present in final dataset; "
               "reporting share of total and of persisters."]
})

# =======================
# Write Excel (1 workbook)
# =======================
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    write_sheet(writer, "Persistence rate", tbl_persistence)
    write_sheet(writer, "Gender diff (tables)", tbl_gender)
    # put z-test on a second table in same sheet (rows after)
    tbl_gender.to_excel(writer, sheet_name="Gender diff (tables)", index=False)
    ws = writer.sheets["Gender diff (tables)"]
    start_row = len(tbl_gender) + 3
    ws.write_string(start_row, 0, "Two-proportion z-test (Male vs Female)")
    ztest_table.to_excel(writer, sheet_name="Gender diff (tables)", index=False, startrow=start_row+1)

    write_sheet(writer, "Race makeup", race_counts)
    write_sheet(writer, "Gender makeup", gender_counts)
    write_sheet(writer, "Race x Gender (counts)", race_gender_counts)
    write_sheet(writer, "Race x Gender (pct)", race_gender_pct)

    write_sheet(writer, "Age distribution", age_summary, image_path=age_plot, image_cell="H2", img_scale=1.0)

    write_sheet(writer, "GPA distribution", gpa_summary, image_path=gpa_plot, image_cell="H2", img_scale=1.0)
    # Also include the GPA counts table on the same sheet below
    gpa_summary.to_excel(writer, sheet_name="GPA distribution", index=False)
    ws = writer.sheets["GPA distribution"]
    start_row = len(gpa_summary) + 3
    ws.write_string(start_row, 0, "GPA counts")
    gpa_counts.to_excel(writer, sheet_name="GPA distribution", index=False, startrow=start_row+1)

    write_sheet(writer, "Avg GPA by program", gpa_by_program)

    write_sheet(writer, "Degree grade distributions", degree_bin_counts, image_path=degree_plot, image_cell="J2", img_scale=1.0)

    write_sheet(writer, "Undecl→Decl proportion", undecl_tbl)

print(f"[OK] Report written: {out_xlsx}")
print("Charts saved to:")
print(" -", age_plot)
print(" -", gpa_plot)
print(" -", degree_plot)

#Including code to not save graphs as well just csv with all content
import os
for img in [age_plot, gpa_plot, degree_plot]:
    if os.path.exists(img):
        os.remove(img)

