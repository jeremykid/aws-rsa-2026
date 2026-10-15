# Bias Audit Report — Diabetic Readmission Dataset

**Date:** 2026-10-14  
**Dataset:** `diabetic/data/diabetic_readmission.csv`  
**Records:** 69,570 &nbsp;|&nbsp; **Columns:** 16  
**Outcome measured:** Any hospital readmission (`<30` days or `>30` days vs. `no`)  
**Overall readmission rate:** 40.2 %

---

## 1. Representation

How many records belong to each demographic group?

| Demographic | Group | Records | Share | Flagged |
|---|---|---:|---:|:---:|
| Race | Caucasian | 53,491 | 76.9 % | |
| Race | African American | 12,887 | 18.5 % | |
| Race | Hispanic | 1,517 | 2.2 % | |
| Race | Other | 1,178 | 1.7 % | |
| Race | **Asian** | **497** | **0.7 %** | ⚠ |
| Gender | Female | 37,044 | 53.2 % | |
| Gender | Male | 32,525 | 46.8 % | |
| Gender | **Unknown / Invalid** | **1** | **< 0.1 %** | ⚠ |
| Age | [70–80) | 17,722 | 25.5 % | |
| Age | [60–70) | 15,522 | 22.3 % | |
| Age | [50–60) | 12,135 | 17.4 % | |
| Age | [80–90) | 11,275 | 16.2 % | |
| Age | [40–50) | 6,681 | 9.6 % | |
| Age | [30–40) | 2,626 | 3.8 % | |
| Age | [90–100) | 1,840 | 2.6 % | |
| Age | [20–30) | 1,090 | 1.6 % | |
| Age | **[10–20)** | **526** | **0.8 %** | ⚠ |
| Age | **[0–10)** | **153** | **0.2 %** | ⚠ |

**Flagged groups** (fewer than 1 000 records): Asian, Unknown/Invalid gender, ages [10–20) and [0–10).  
Any statistics for these groups are based on very few patients and should be interpreted with caution.

**Chart:** `charts/bias_01_representation.png`

---

## 2. Outcome Disparities

Readmission rates are compared to the overall rate of **40.2 %**. Groups deviating by more than **5 percentage points** are flagged.

### By Race

| Group | Records | Readmission rate | Deviation | Flagged |
|---|---:|---:|---:|:---:|
| Caucasian | 53,491 | 41.0 % | +0.8 pp | |
| African American | 12,887 | 38.5 % | −1.7 pp | |
| Hispanic | 1,517 | 35.3 % | −4.9 pp | |
| **Other** | **1,178** | **33.2 %** | **−7.0 pp** | ⚠ |
| **Asian** | **497** | **29.8 %** | **−10.4 pp** | ⚠ |

Asian and Other patients show meaningfully lower readmission rates than Caucasian patients. Because the Asian group is also small (497 records), the estimate is uncertain. The lower rate may reflect true clinical differences, care-seeking behaviour, or under-representation of higher-risk patients in this sample.

### By Gender

| Group | Records | Readmission rate | Deviation | Flagged |
|---|---:|---:|---:|:---:|
| Female | 37,044 | 41.0 % | +0.8 pp | |
| Male | 32,525 | 39.3 % | −0.9 pp | |
| **Unknown / Invalid** | **1** | **0.0 %** | **−40.2 pp** | ⚠ |

The single `Unknown/Invalid` record is statistically meaningless. Male and female rates are nearly identical — no meaningful gender disparity in outcomes.

### By Age Group

| Group | Records | Readmission rate | Deviation | Flagged |
|---|---:|---:|---:|:---:|
| **[0–10)** | **153** | **17.6 %** | **−22.5 pp** | ⚠ |
| **[10–20)** | **526** | **31.6 %** | **−8.6 pp** | ⚠ |
| **[20–30)** | **1,090** | **32.3 %** | **−7.9 pp** | ⚠ |
| **[30–40)** | **2,626** | **32.9 %** | **−7.3 pp** | ⚠ |
| [40–50) | 6,681 | 36.0 % | −4.2 pp | |
| [50–60) | 12,135 | 37.3 % | −2.9 pp | |
| [60–70) | 15,522 | 40.2 % | +0.0 pp | |
| [70–80) | 17,722 | 44.0 % | +3.8 pp | |
| [80–90) | 11,275 | 43.9 % | +3.7 pp | |
| **[90–100)** | **1,840** | **34.1 %** | **−6.1 pp** | ⚠ |

A clear age gradient exists: readmission rates rise steadily from young adults through the 70s–80s, then dip in the oldest patients. Younger age groups (under 40) have substantially lower rates. This likely reflects both true clinical differences (younger diabetic patients may have less severe disease) and small sample sizes.

**Chart:** `charts/bias_02_outcomes.png`

---

## 3. Missing Data

After checking for both `NaN` and `?` sentinel values across all 16 columns, **no missing data was found** in this dataset. Missingness analysis by demographic group is therefore not applicable.

No chart was generated for this section.

---

## Summary of Flags

| Check | Flagged group | Issue |
|---|---|---|
| Representation | Asian (race) | Only 497 records (0.7 %) |
| Representation | Unknown/Invalid (gender) | Only 1 record |
| Representation | [10–20) (age) | Only 526 records |
| Representation | [0–10) (age) | Only 153 records |
| Outcomes | Asian | −10.4 pp vs. overall |
| Outcomes | Other (race) | −7.0 pp vs. overall |
| Outcomes | Unknown/Invalid (gender) | −40.2 pp (single record, ignore) |
| Outcomes | Ages [0–10) through [30–40) | −7 to −23 pp vs. overall |
| Outcomes | [90–100) (age) | −6.1 pp vs. overall |

---

## Limitations

1. **Observational data.** Differences in readmission rates across groups reflect a mixture of disease severity, social determinants, access to care, and coding practices — not necessarily care quality alone.
2. **Small groups are unreliable.** Estimates for Asian, pediatric, and young-adult groups are based on few patients and have wide confidence intervals. Conclusions about these groups should not be drawn without additional data.
3. **Race/ethnicity is self-reported and coarse.** The five race categories mask substantial within-group heterogeneity.
4. **Binary outcome.** Treating `<30` and `>30` readmissions identically collapses an important clinical distinction (urgent vs. planned readmission).
5. **No confounding adjustment.** The comparisons are unadjusted. Age differences in outcome rates, for example, may partly reflect differences in disease duration or comorbidity burden rather than age itself.
6. **Single institution / time window.** Findings may not generalise to other hospitals or time periods.
