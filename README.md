# TRIAGEGEIST: A Triage Second-Opinion System

**Clinical AI for Undertriage Detection, NLP Ablation, Demographic Equity Audit, and Waiting Room Risk Stratification**

Kaggle Competition: [Triagegeist](https://www.kaggle.com/competitions/triagegeist)  
Writeup: [kaggle.com/competitions/triagegeist/writeups/triagegeist-a-triage-second-opinion-system](https://www.kaggle.com/competitions/triagegeist/writeups/triagegeist-a-triage-second-opinion-system)

---

## Overview

TRIAGEGEIST is a clinical AI second-opinion system for emergency department triage. It extends the federated undertriage reduction framework of Laitinen-Imanov & Dulger (2026) with four contributions:

1. **Quantified demographic neutrality** — QWK gap = −0.0003, with a proposed audit threshold of 0.005 for real-world deployment
2. **NLP ablation study** — Keyword (+0.0045) vs LSA (+0.0693) contributions isolated via 5-model architecture
3. **Elderly × silent pain interaction** — 11–12% undertriage rate (4.6x population average), with a three-filter triage rule
4. **Waiting Room Deterioration Risk System (WRRS)** — 4-tier re-assessment queue deployed on training and unseen test patients

---

## Key Results

| Metric | Value |
|--------|-------|
| Clinical model QWK (no demographics) | 0.9296 |
| Full model QWK (with demographics) | 0.9293 |
| QWK gap (demographic influence) | −0.0003 |
| NLP-Enhanced model QWK | 0.9989 |
| Ensemble QWK (5 models) | 0.9988 |
| Undertriaged patients | 5,484 (6.9%) |
| Elderly × no pain undertriage rate | 11.4% (4.6x average) |
| Age group disparity p-value | 0.0001 |
| WRRS RED-tier test patients | 86 (mean NEWS2: 13.7) |

---

## Model Architecture

| Model | Features | CV QWK |
|-------|----------|--------|
| A — Clinical | Vitals, NEWS2, GCS, comorbidities (no demographics) | 0.9296 |
| B — Full | Model A + demographics | 0.9293 |
| C — NLP-Enhanced | Model A + 15 keywords + 20 LSA components | 0.9989 |
| D — Keyword-Only | Model A + keywords only (ablation) | 0.9341 |
| E — LSA-Only | Model A + LSA only (ablation) | 0.9989 |
| Ensemble | Optimized blend of A–E | 0.9988 |

---

## Files

| File | Description |
|------|-------------|
| `triagegeist_kaggle.py` | Main solution: 5-model ensemble, NLP ablation, WRRS, equity audit |
| `triagegeist_kaggle.ipynb` | Kaggle Notebook version |
| `writeup_v4.md` | **Full writeup (v4 — current submission version)** |
| `writeup_paste_v4.txt` | Plain-text paste version of the v4 writeup |
| `cover_image_560x280.png` | Cover image (Kaggle-spec 560×280) |
| `fig1_overview.png` | Main 8-panel analysis figure |
| `fig2_decision_support.png` | Alert system visualization |
| `fig3_provider_analysis.png` | Provider-level undertriage audit |
| `fig4_nlp_analysis.png` | NLP keyword undertriage enrichment |
| `fig5_wrrs_dashboard.png` | WRRS tier dashboard |
| `fig6_ablation_intersectional.png` | NLP ablation + intersectional heatmap + WRRS tier comparison |
| `cover_image_v2.png` | Thumbnail image |

---

## Setup

```bash
pip install lightgbm shap scikit-learn pandas numpy matplotlib seaborn
python triagegeist_kaggle.py
```

Data files (`train.csv`, `test.csv`, `chief_complaints.csv`, `patient_history.csv`) must be placed in the same directory or at `/kaggle/input/competitions/triagegeist/`. Available from the [Kaggle competition page](https://www.kaggle.com/competitions/triagegeist/data).

Runtime: ~90–100 minutes on Kaggle CPU.

---

## NLP Ablation Results

| Model | CV QWK | Lift vs Clinical |
|-------|--------|-----------------|
| Clinical baseline | 0.9296 | — |
| + Keywords only | 0.9341 | +0.0045 |
| + LSA only | 0.9989 | +0.0693 |
| + Keywords + LSA | 0.9989 | +0.0693 |

> **Note:** The large LSA lift reflects synthetic data properties (text-acuity alignment). A text-shuffle control experiment is recommended before accepting this lift in real-world deployment.

---

## WRRS (Waiting Room Deterioration Risk System)

Composite 0–100 score integrating NEWS2 (35 pts), shock index (20 pts), GCS (15 pts), elderly status (10 pts), NLP keywords (10 pts), triage gap (10 pts).

| Tier | Re-assess | Training undertriaged | Test (all patients) |
|------|-----------|-----------------------|---------------------|
| RED | ≤5 min | 28 (0.5%) | 86 (0.4%) |
| ORANGE | ≤15 min | 140 (2.6%) | 2,482 (12.4%) |
| YELLOW | ≤30 min | 226 (4.1%) | 2,362 (11.8%) |
| GREEN | ≤60 min | 5,090 (92.8%) | 15,070 (75.3%) |

---

## References

- Laitinen-Imanov GO, Dulger SB. (2026). Federated Multimodal Learning with Differential Privacy for Emergency Department Triage. SSRN 6282898.
- Gilboy N, et al. (2012). Emergency Severity Index (ESI). AHRQ Publication No. 12-0014.
- Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. NeurIPS 30.
- Smith GB, et al. (2013). National Early Warning Score (NEWS). Resuscitation, 84(4), 465–470.
- Platts-Mills TF, et al. (2012). Inadequate analgesia in elderly patients. J Emergency Medicine, 42(1), 13–19.
