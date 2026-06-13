# TRIAGEGEIST: A Triage Second-Opinion System

**Detecting Undertriage Risk via Clinical AI and Demographic Equity Analysis**

Kaggle Competition: [Triagegeist](https://www.kaggle.com/competitions/triagegeist)

---

## Overview

This project builds an objective clinical model that detects systematic undertriage in emergency departments — patients whose assigned triage acuity is lower than their physiological data warrants.

**Key Findings:**
- Clinical model achieves QWK = 0.929 using vital signs alone (no demographics)
- 6.9% of patients are undertriaged by at least one ESI level
- Elderly patients reporting no/mild pain have 11-12% undertriage rate (nearly 2x average)
- Age group disparity is statistically significant (chi-square p = 0.0001)
- SHAP identifies pain_score, GCS, and NEWS2 as top clinical drivers

## Files

| File | Description |
|------|-------------|
| `triagegeist_final.py` | Main solution: dual-model design, SHAP, equity audit, provider analysis |
| `writeup_final.md` | Full project writeup (clinical problem, methodology, findings, limitations) |
| `fig1_overview.png` | Main 8-panel analysis figure |
| `fig2_decision_support.png` | Alert system visualization |
| `fig3_provider_analysis.png` | Provider-level (nurse/site) undertriage audit |

## Setup

```bash
pip install lightgbm shap scikit-learn pandas numpy matplotlib seaborn
python triagegeist_final.py
```

Data files (`train.csv`, `test.csv`, `chief_complaints.csv`, `patient_history.csv`) must be placed in the same directory. Available from the [Kaggle competition page](https://www.kaggle.com/competitions/triagegeist/data).

## Approach

1. **Dual-model design**: Clinical model (no demographics) vs Full model — QWK gap measures demographic influence
2. **Triage gap alert**: Flags patients where clinical model predicts >= 1 level more urgently than assigned acuity
3. **SHAP explanations**: Feature-level explanations for individual undertriage alerts
4. **Demographic equity audit**: Chi-square tests across sex, language, insurance, age group
5. **Provider analysis**: Nurse and site-level undertriage rate monitoring

## Results

| Metric | Value |
|--------|-------|
| Clinical model QWK | 0.9296 |
| Full model QWK | 0.9293 |
| Undertriaged patients | 5,484 (6.9%) |
| Elderly × no pain undertriage | 11.4% |
| Age group p-value | 0.0001 |
