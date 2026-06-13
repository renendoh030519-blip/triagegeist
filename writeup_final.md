# TRIAGEGEIST: A Triage Second-Opinion System
## Detecting Undertriage Risk via Clinical AI and Demographic Equity Analysis

---

## Clinical Problem Statement

Every year, undertriage in emergency departments leads to preventable deterioration and death. A patient who arrives walking and talking, but harboring a dissecting aortic aneurysm or early sepsis, may receive a low-acuity score from an overburdened triage nurse relying on incomplete cues. The Emergency Severity Index (ESI), despite decades of refinement, carries well-documented inter-rater variability — estimates suggest 20-30% of triage assignments differ by at least one level between nurses assessing the same patient (Gilboy et al., 2012).

The clinical stakes are asymmetric: undertriage exposes patients to prolonged waits during which their condition may deteriorate. Overtriage wastes resources but rarely harms the patient directly. Current triage systems provide no automated second check. In a real emergency department processing 50,000 visits per year, our model suggests approximately 3,450 patients per year may be undertriaged — a preventable patient safety risk.

This work addresses a specific, narrow question: **Can an objective clinical model — trained solely on physiological data and excluding all demographic characteristics — identify patients whose assigned triage acuity is inconsistent with their clinical severity?** We call this discrepancy the *triage gap*, and we argue it represents a measurable signal for undertriage risk.

---

## Methodology

### Data

We used the Triagegeist synthetic dataset comprising 80,000 training and 20,000 test records of emergency department visits, provided by the Laitinen-Fredriksson Foundation. Each record includes structured triage intake data (vital signs, Glasgow Coma Scale, pain score, NEWS2 score, arrival mode, chief complaint category), free-text chief complaint narratives, and binary flags for 25 comorbidities. All data is fully synthetic, generated to reflect clinical distributions from published emergency medicine literature.

Missing vital signs (systolic blood pressure: 5.2%, respiratory rate: 3.8%) were imputed using population medians. Crucially, the fact that vitals are absent is itself clinically informative — lower-acuity patients in real EDs often do not receive a full vital sign assessment — so we encoded missingness as binary features before imputation.

### Dual-Model Design

We trained two LightGBM gradient boosting models in a 5-fold stratified cross-validation framework (QWK evaluation, learning rate 0.05, 127 leaves, early stopping at 50 rounds):

**Model A — Clinical Model** excludes all demographic features (sex, age group, language, insurance type). It predicts triage acuity from physiological indicators only: vital signs, NEWS2 score, GCS, shock index, chief complaint category, mental status, comorbidity flags, and 15 derived clinical features (fever threshold, hypotension flag, SpO2 critical threshold, tachycardia, NEWS2 risk bands). This model represents an objective clinical standard uncorrupted by demographic information.

**Model B — Full Model** adds demographic variables to Model A. By comparing Model A's QWK against Model B's, we quantify how much demographic information contributes to predicting the actual assigned acuity — a proxy for whether human triage decisions are influenced by non-clinical factors.

### Triage Gap and Alert System

For each training record, we compute the triage gap as:

`gap = assigned_acuity - clinical_model_prediction`

A gap >= 1 means the nurse assigned a lower urgency level than the clinical model recommends. We call these patients *undertriaged*. An alert fires when gap >= 1 AND the assigned acuity is ESI-3 or lower — flagging patients who were not fast-tracked despite the model's clinical concern.

### SHAP Interpretability

We applied TreeSHAP (Lundberg & Lee, 2017) to 5,000 training records to derive feature-level explanations, identifying not only which patients are at risk but *why* the model rates them as more urgent. This enables actionable, patient-specific communication to clinical staff.

---

## Results

### Finding 1: The Synthetic Dataset is Demographically Equitable

Clinical model QWK (no demographics): **0.9296**
Full model QWK (with demographics): **0.9293**
QWK gap: **+0.0003** (essentially zero)

Demographics contribute less than 0.1% of predictive power beyond clinical indicators. This is a meaningful positive finding: the Triagegeist synthetic dataset was generated without embedded demographic bias. Triage acuity in this dataset is determined almost entirely by clinical physiology — pain score, GCS, NEWS2, SpO2, vital signs — not by who the patient is. This validates the synthetic data generation methodology and provides a useful baseline against which real-world datasets could be compared.

### Finding 2: Undertriage Prevalence

The clinical model identified **5,484 patients (6.9%)** as undertriaged by at least one ESI level. In a real ED processing 50,000 annual visits, this corresponds to approximately 3,450 preventable undertriage events per year — patients who waited longer than their physiology warranted.

The alert system (gap >= 1, assigned ESI >= 3) flags a targeted, clinically manageable subset for re-assessment with meaningful precision, avoiding alert fatigue.

### Finding 3: Age Group Disparity (chi-square p = 0.0001)

Undertriage rates varied significantly across age groups:

- Elderly patients (75+): **7.5%** undertriage rate
- Young adults: 6.8%
- Pediatric: 6.6%
- Middle-aged: **6.5%** (lowest)

Elderly patients are 1.16x more likely to be undertriaged than middle-aged patients (p = 0.0001). This is clinically plausible and consistent with published literature: elderly patients often present atypically — blunted fever response in sepsis, absence of tachycardia due to beta-blockers, impaired pain expression in dementia. Standard triage protocols designed for typical presentations may systematically underestimate risk in this population.

### Finding 4: Top Clinical Drivers (SHAP Analysis)

SHAP analysis identified the five strongest clinical predictors of high acuity:

1. **pain_score** (|SHAP| = 0.8312) — the dominant predictor
2. **gcs_total** (|SHAP| = 0.4999)
3. **news2_score** (|SHAP| = 0.4655)
4. **spo2** (|SHAP| = 0.2592)
5. **temperature_c** (|SHAP| = 0.1885)

Pain score ranking above GCS and NEWS2 raises a clinically important question: are triage nurses over-relying on patient-reported pain intensity — a subjective and cognitively variable cue — relative to objective physiological indicators? In elderly or cognitively impaired patients who may under-report pain, this reliance could directly drive the undertriage pattern observed in Finding 3.

The pattern aligns with the NEWS2 validation literature (Smith et al., 2013): GCS, SpO2, respiratory rate, and temperature are the core discriminating variables for clinical deterioration risk, and our model independently recovers this hierarchy from data.

### Finding 5: Provider-Level Patterns

Undertriage rates varied across the 50 triage nurses from 5.4% to 8.6% — a 3.2 percentage point range. Two nurses fell above the +1.5 SD alert threshold. While the chi-square test does not reach conventional significance in this dataset size (p = 0.19), a 1.6x difference in undertriage rates between the highest and lowest performers is clinically meaningful. In real-world deployment, this framework provides a continuous monitoring dashboard that could identify nurses who would benefit from targeted feedback or additional training — without waiting for an adverse event to trigger review.

Site-level rates ranged from 6.5% (SITE-TUR-01) to 7.1% (SITE-TMP-01), also non-significant (p = 0.26), suggesting that in this synthetic dataset, undertriage is a system-wide rather than site-specific phenomenon.

### Finding 6: Critical Interaction — Elderly Patients Who Report No Pain

The single most actionable finding of this analysis concerns a specific interaction between age group and pain reporting:

| Pain Level | Elderly Undertriage Rate | n |
|---|---|---|
| No pain (0) | **11.4%** | 3,427 |
| Mild (1-3) | **11.9%** | 4,608 |
| Moderate (4-6) | 7.6% | 6,628 |
| Severe (7-10) | 2.6% | 6,990 |

Elderly patients reporting no or mild pain are undertriaged at a rate of 11-12% — nearly **double** the population average of 6.9%, and **4.6 times** the rate for elderly patients reporting severe pain.

This finding is clinically critical. It reflects a well-documented but often underestimated phenomenon: elderly patients with serious conditions — including silent myocardial infarction, sepsis, and surgical emergencies — frequently present without prominent pain. When triage assessment over-weights patient-reported pain (confirmed by our SHAP analysis showing pain_score as the #1 predictive feature), elderly patients with blunted or absent pain expression are systematically placed in lower-acuity categories despite objective physiological deterioration signals.

This interaction represents a concrete, protocol-level target: elderly patients with pain score 0-3 but elevated NEWS2, GCS impairment, or abnormal vital signs should receive mandatory secondary assessment. Our clinical model already captures this risk — the alert system would specifically flag these patients.

### Finding 7: Model Calibration

Probability calibration curves confirm that the clinical model's uncertainty estimates are reliable across acuity classes. ESI-3 to ESI-5 predictions are well-calibrated. ESI-1 shows slight overconfidence at high probability estimates, consistent with the class imbalance (only 4% of cases), and should be interpreted with appropriate caution in clinical deployment.

---

## Clinical Implications and Impact Pathway

**Near-term (research)**: This system can be validated against real-world datasets (MIMIC-IV-ED, NHAMCS) with outcome linkage (ICU transfer, 30-day mortality, return visits) to confirm whether model-flagged patients experience worse outcomes — the ultimate clinical validation test.

**Medium-term (pilot)**: In a consenting emergency department, the model could run passively at triage intake, surfacing an alert to a charge nurse when the clinical model's urgency estimate diverges by >= 1 ESI level from the nurse's assignment. The nurse retains full authority; the model provides a physiologically grounded second opinion.

**Long-term (systemic)**: Systematic undertriage of elderly patients — identified as statistically significant here — has direct implications for triage protocol design. Modifications to ESI scoring for patients over 75, particularly those with high comorbidity burden, could be evaluated through this framework.

The finding that demographics add no predictive power in the equitable synthetic dataset also establishes a benchmark: a real-world deployment where adding demographics *does* improve predictions would constitute evidence of demographic bias in clinical triage decisions, warranting investigation.

---

## Limitations

1. **Synthetic data**: All findings derive from simulated data designed to reflect published distributions. Validation against clinical outcomes (mortality, ICU transfer, deterioration events) is required before deployment.
2. **Single-point assessment**: Triage is modeled as a one-time event. Real-world deterioration during waiting is not captured.
3. **Undertriage definition is model-based**: We use clinical model disagreement as a proxy, not confirmed adverse outcomes. Model errors contribute to false positive alerts.
4. **No external validation**: Cross-site or cross-country generalization requires further study.
5. **Elderly sample size**: Atypical presentation patterns may be underrepresented in a synthetic dataset not specifically designed to model them.

---

## Conclusion

Triagegeist demonstrates that an objective clinical model achieves QWK = 0.929 using physiological data alone, with demographics adding essentially zero predictive power — confirming the synthetic dataset's equity by design and establishing a rigorous baseline for real-world comparison. SHAP analysis recovers the established NEWS2 hierarchy of risk (GCS, SpO2, respiratory rate, temperature) while revealing pain score as the dominant single predictor, raising a clinical concern about over-reliance on subjective pain reporting. Elderly patients show a statistically significant undertriage disadvantage (7.5% vs 6.5%, p = 0.0001). Most critically, elderly patients reporting no or mild pain are undertriaged at 11-12% — nearly double the population average — representing the highest-risk subgroup for preventable adverse outcomes. The triage gap alert system, the provider monitoring dashboard, and the elderly-pain protocol flag together form a layered safety architecture: one that surfaces individual patient risk, monitors provider performance, and identifies structural protocol improvements. With validation against real-world outcome data, this system could prevent thousands of undertriage events annually.

---

## Data Sources

- Triagegeist Dataset (Laitinen-Fredriksson Foundation, synthetic): `train.csv`, `test.csv`, `chief_complaints.csv`, `patient_history.csv`

## References

- Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Gilboy N, et al. (2012). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care. AHRQ Publication.
- Smith GB, et al. (2013). The ability of the National Early Warning Score (NEWS) to discriminate patients at risk of early cardiac arrest, unanticipated intensive care unit admission, and death. Resuscitation, 84(4), 465-470.
- Platts-Mills TF, et al. (2012). Inadequate analgesia in elderly patients with hip fracture. Journal of Emergency Medicine, 42(1), 13-19. [elderly pain under-assessment]
