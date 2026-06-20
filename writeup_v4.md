# TRIAGEGEIST: A Triage Second-Opinion System
## Clinical AI for Undertriage Detection, Demographic Equity Analysis, and Waiting Room Risk Stratification

> **At a Glance** — A triage second-opinion system that catches the patients standard triage misses: elderly patients with silent pain are undertriaged at 11.4–11.9% (4.6× the population average), and a simple three-filter rule flags ~350 such high-risk cases/year at a 3:1 false-alert ratio in a 50,000-visit ED. Built on a 5-model ensemble (80,000 train + 20,000 test), with demographic neutrality (QWK gap = −0.0003), a deployable WRRS (86 RED-tier test patients), and a text-shuffle leakage control confirming the NLP lift is genuine (QWK 0.9989 → 0.9290 when permuted).

---

## Why This Work Is New

Most triage AI submissions stop at accuracy. TRIAGEGEIST adds three things that are absent from existing work: (1) a **text-shuffle leakage control** — we actively test and confirm the NLP lift is genuine, not a synthetic data artifact; (2) a **quantified demographic audit threshold** (QWK gap 0.005) that operationalizes "no bias" into a monitorable deployment gate; and (3) a **WRRS deployed on unseen test patients without ground-truth labels**, using inter-model disagreement as an uncertainty proxy — converting a research model into a deployable waiting-room tool.

---

## Clinical Problem Statement

Emergency department undertriage — assigning lower acuity than clinical conditions warrant — causes preventable deterioration and death. The Emergency Severity Index carries documented inter-rater variability of 20–30% across nurses assessing identical patients (Gilboy et al., 2012). Laitinen-Imanov & Dulger (2026) demonstrated that federated ML can reduce overall mistriage from 8.5% to 3.8% (p < 0.001) across a multi-site ED network. TRIAGEGEIST extends this framework with four contributions: (1) a demographic equity audit with a proposed real-world deployment threshold; (2) an NLP ablation pipeline with leakage control; (3) a Waiting Room Deterioration Risk System (WRRS) deployed on unseen test patients; and (4) a three-filter triage protocol rule with quantified clinical impact.

---

## Methodology

**Data:** The Triagegeist competition's provided multi-site synthetic ED dataset — 80,000 training and 20,000 test records (distinct from the 87,234-encounter `fedmml-ed-triage` set of the reference paper; here the federated framework is cited for clinical context only). Each record carries structured vitals, GCS, NEWS2, free-text chief complaints, 25 comorbidity flags, and patient history. Missing vitals (BP: 5.2%, RR: 3.8%) were median-imputed; missingness preserved as binary indicator features.

**Model Architecture:** Five LightGBM models in 5-fold stratified CV (QWK, early stopping at 50 rounds):

| Model | Features | Purpose |
|-------|----------|---------|
| A — Clinical | Vitals, NEWS2, GCS, comorbidities | Bias-free clinical standard |
| B — Full | Model A + sex, age, language, insurance | Demographic influence measurement |
| C — NLP-Enhanced | Model A + 15 keywords + 20 LSA components | Full text-augmented prediction |
| D — Keyword-Only | Model A + 15 keywords | Ablation: keyword contribution |
| E — LSA-Only | Model A + 20 LSA components | Ablation: semantic contribution |

Ensemble: grid-search blend of all five models on out-of-fold predictions.

**NLP Pipeline:** (1) 15 clinical keyword flags (sepsis, stroke, bleeding, chest pain, altered mental status, etc.) via case-insensitive matching. (2) TF-IDF (1,000 features, bigrams, sublinear TF) + TruncatedSVD (20 components, 18.5% variance explained) generating dense semantic features.

**WRRS:** Composite 0–100 score — NEWS2 (35 pts), shock index (20 pts), GCS impairment (15 pts), elderly status (10 pts), NLP keywords (10 pts), triage gap/model uncertainty (10 pts). Tiers: RED (≥75, ≤5 min), ORANGE (≥55, ≤15 min), YELLOW (≥35, ≤30 min), GREEN (≤60 min). On unseen test patients, inter-model disagreement (|NLP − Clinical prediction|) replaces the unavailable triage gap as an uncertainty proxy.

---

## Results

**Finding 1 — Demographic Neutrality and Equity Audit:** Clinical QWK = 0.9296, Full (with demographics) = 0.9293, gap = −0.0003. Demographic features account for <0.03% of predictive power. We propose **0.005 as a real-world audit threshold** — detectable at p < 0.01 with 80,000 records, and 10× our baseline. Intersectionally, elderly undertriage rates are 7.3–8.95% across all sex groups vs 6.5–6.8% in younger cohorts, confirming age as the primary equity risk factor independent of sex or insurance type. Fairness here is therefore two-sided: direct demographic parity across sex, language, and insurance holds, while an intersectional inequity (age × silent pain) persists and is the actionable target. (See Figure 1 for age/language breakdown and undertriage funnel; Figure 6 Panel B for intersectional heatmap.)

**Finding 2 — Elderly × Silent Pain Interaction (primary clinical finding):** 5,484 patients (6.9%) were undertriaged overall — a rate comparable to the 8.5% pre-intervention *overall mistriage* (over- and under-triage combined) reported by Laitinen-Imanov & Dulger (2026). The critical subgroup:

| Pain Level | Elderly Undertriage Rate | n |
|-----------|------------------------|---|
| No pain (0) | **11.4%** | 3,427 |
| Mild (1–3) | **11.9%** | 4,608 |
| Moderate (4–6) | 7.6% | 6,628 |
| Severe (7–10) | 2.6% | 6,990 |

11–12% rate = 4.6× population average. SHAP analysis explains the mechanism: pain_score dominates prediction (|SHAP| = 0.831, vs news2_score = 0.466), consistent with subjective pain being over-weighted relative to objective deterioration in this synthetic data — systematically under-triaging elderly patients who under-report it. A three-filter protocol rule (age ≥75 + pain ≤3 + NEWS2 ≥3 or GCS ≤13 or NLP keyword) yields ~350 caught cases/year at 3:1 false-alert ratio in a 50,000-visit ED. (See Figure 1 for SHAP feature importance; Figure 3 for elderly×pain interaction chart; Figure 2 for alert system visualization.)

**Finding 3 — NLP Ablation:**

| Model | CV QWK | Lift vs Clinical |
|-------|--------|-----------------|
| Clinical baseline | 0.9296 | — |
| + Keywords only | 0.9341 | +0.0045 |
| + LSA only | 0.9989 | +0.0693 |
| + Keywords + LSA | 0.9989 | +0.0693 |

Both components contribute independently; LSA's semantic signal, however, saturates the achievable QWK, so adding keywords on top yields no further lift (0.9989 → 0.9989). The keyword-only path (+0.0045) therefore remains the more transferable real-world signal. Bleeding shows the highest undertriage enrichment (1.26×, p = 0.029), consistent with early hemorrhage where vital compensation temporarily masks instability. The LSA lift magnitude is validated by the text-shuffle control (see below). (See Figure 4 for keyword undertriage enrichment; Figure 6 Panel A for ablation comparison chart.)

**Finding 4 — WRRS Validation on Unseen Test Patients:**

| Tier | Training (UT patients) | NEWS2 | Elderly % | Test (all 20,000) |
|------|----------------------|-------|-----------|-------------------|
| RED | 28 (0.5%) | 13.4 | 71% | 86 (0.4%) |
| ORANGE | 140 (2.6%) | 10.8 | 27% | 2,482 (12.4%) |
| YELLOW | 226 (4.1%) | 4.2 | 43% | 2,362 (11.8%) |
| GREEN | 5,090 (92.8%) | 1.0 | 29% | 15,070 (75.3%) |

Training-column percentages are computed over the 5,484 undertriaged patients only, whereas test-column percentages span all 20,000 patients; the apparent ORANGE shift (2.6% → 12.4%) therefore reflects these different denominators, not tier instability. Test RED-tier patients (no ground-truth labels available): mean NEWS2 = 13.7, GCS = 5.8 vs overall 14.2. Physiological validity confirmed on unseen data using inter-model disagreement as uncertainty proxy. Scaling the training tier counts among undertriaged patients (28 RED, 140 ORANGE) to a 50,000-visit ED yields ~17 RED + 87 ORANGE *second-opinion escalations among undertriaged patients* per year — a manageable load, and distinct from the full-population tier volumes in the Test column (which span all patients). (See Figure 5 for WRRS tier dashboard; Figure 6 Panel C for training vs test tier comparison.)

**Finding 5 — Ensemble and Provider Audit:** Optimized blend (NLP=0.50, LSA=0.21, Clin=0.10, KW=0.10, Full=0.09) achieves ensemble QWK = 0.9988, near-parity with the best single model (0.9989). Separately, provider-level monitoring detected a 5.4%–8.6% undertriage spread across 50 nurses (1.6× range); 2 exceeded +1.5 SD threshold — actionable via targeted feedback without systemic protocol revision. (See Figure 3 for provider-level undertriage rate chart.)

---

## NLP Signal Validity: Text-Shuffle Control

The near-perfect LSA QWK (0.9989) requires active leakage testing, not passive disclaimer. We ran a text-shuffle control (Section 6d of the accompanying code): randomly permute `chief_complaint_raw` across all 80,000 patients, re-fit TF-IDF + SVD on permuted text, retrain LSA-Only model in 5-fold CV.

| Condition | LSA-Only CV QWK |
|-----------|----------------|
| Genuine text | 0.9989 |
| Shuffled text (permuted) | **0.9290** |
| Clinical baseline (no text) | 0.9296 |

**Verdict: DROPS TO BASELINE.** QWK collapses to clinical baseline when text is permuted, confirming the model learns from genuine patient-level text-acuity correlations — not corpus-level leakage or pipeline artifacts. In real ED data, text-acuity correlations persist but lift magnitude will be lower due to noisier free text; the keyword component (+0.0045) provides a more conservative real-world lower bound. For calibration, published real-world ED triage models typically report QWK ≈ 0.5–0.7; our 0.9989 reflects this synthetic corpus's clean text–acuity alignment rather than an expected deployment figure. Code is included so any reviewer can independently verify the verdict.

---

## Clinical Implications

TRIAGEGEIST is designed as the interpretability and risk-stratification layer of the Laitinen-Imanov & Dulger (2026) federated framework. Their system achieves site-level model aggregation with differential privacy but does not address per-patient actionability; TRIAGEGEIST fills this gap with a WRRS re-assessment queue, demographic fairness gating (QWK gap threshold: 0.005), and nurse-level monitoring. In federated deployment, the demographic audit threshold serves as a site-level fairness gate before weight aggregation. Validation path: MIMIC-IV-ED outcome linkage → 2–3 site federated pilot → three-filter rule integration into clinical triage software (Epic/Cerner).

---

## Limitations

1. LSA lift magnitude reflects this synthetic dataset's text-acuity generation process; real-world lift will be lower.
2. Undertriage is defined by model-clinical disagreement, not confirmed adverse outcomes; ICU/mortality linkage required for clinical validation.
3. WRRS weights are literature-derived, not empirically optimized on deterioration outcomes.
4. Differential privacy and cross-site aggregation are inherited from Laitinen-Imanov & Dulger (2026), not implemented here.
5. Three-filter rule impact estimates assume training-data prevalence generalizes to the deployment ED.

---

## Conclusion

TRIAGEGEIST moves beyond accuracy reporting to deliver three deployment-ready contributions: a demographic audit gate (QWK gap −0.0003, proposed threshold 0.005), a clinically actionable subgroup finding (elderly × silent pain at 4.6× undertriage rate with a concrete three-filter protocol rule), and a WRRS that runs on unseen patients without ground-truth labels. The text-shuffle control confirms NLP signal validity rather than assuming it. Together these form the interpretability and risk-stratification layer that completes the federated framework of Laitinen-Imanov & Dulger (2026).

---

## Reproducibility Notes

- Notebook: https://www.kaggle.com/code/renendoh/triagegeistww (public, end-to-end including shuffle control, ~100–110 min CPU)
- GitHub: https://github.com/renendoh030519-blip/triagegeist
- Seed: 42 throughout | Packages: lightgbm, shap, scikit-learn, scipy, pandas, numpy

---

## References

- Laitinen-Imanov GO, Dulger SB. (2026). Federated Multimodal Learning with Differential Privacy for Emergency Department Triage. SSRN 6282898.
- Gilboy N, et al. (2012). Emergency Severity Index. AHRQ No. 12-0014.
- Lundberg SM, Lee SI. (2017). Unified approach to interpreting model predictions. NeurIPS 30.
- Smith GB, et al. (2013). National Early Warning Score. Resuscitation, 84(4), 465–470.
- Platts-Mills TF, et al. (2012). Inadequate analgesia in elderly ED patients. J Emergency Medicine, 42(1), 13–19.
- Bayer AJ, et al. (1986). Changing presentation of myocardial infarction with increasing old age. JAGS, 34(4), 263–266.
