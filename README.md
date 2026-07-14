# Low-Burden, History- and Resource-Adaptive MCI-to-AD Risk Prediction

[![Tests](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml/badge.svg)](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)
![Version](https://img.shields.io/badge/model-V6.1--Hybrid--QC-purple)

A clinical-AI research prototype for dynamic **1-, 2-, 3-, and 5-year prediction of progression from mild cognitive impairment (MCI) to Alzheimer’s disease dementia**, using routine cognitive and functional assessments.

**[Open the model-backed demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)**
[Model Card](docs/MODEL_CARD.md) · [Complete Frozen Pipeline](https://github.com/rey-Lii/mci-ad-resource-adaptive-transformerrr)

> Research use only. This retrospective prototype is not intended for clinical diagnosis or patient-level decision-making.

---

## Overview

Clinical histories differ in both depth and assessment availability. Some patients have only one assessment date or irregular longitudinal follow-up, particularly in primary-care, community, or resource-constrained settings. Entire cognitive or functional modules may also be unavailable across patients or cohorts.

To accommodate these differences, the system uses:

* **one assessment date** → regularized Snapshot survival expert;
* **two or more assessment dates** → modular longitudinal Transformer;
* **heterogeneous assessment availability** → explicit module-level availability representation.

The model uses routine clinical assessments without requiring PET, CSF, MRI, or genetic biomarkers.

---

## Highlights

* Dynamic 1-, 2-, 3-, and 5-year MCI-to-AD dementia risk prediction.
* Leakage-aware dynamic MCI landmark construction.
* Deterministic history-adaptive routing.
* Five separately represented cognitive and functional modules.
* Irregular-time features, trajectory summaries, and latest-state anchoring.
* Patient-grouped ADNI development evaluation.
* Frozen zero-shot NACC external evaluation without retraining.
* Public input validation, tensor construction, architecture, routing, tests, and aggregate results.

---

## How the system works

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7aef8a20-1c11-4ba9-a8dd-522bbde6c469"
    alt="Overview of the history- and resource-adaptive hybrid survival system"
    width="82%"
  />
</p>

The longitudinal branch models five assessment modules separately:

* ADAS13
* MMSE
* global CDR
* CDR Sum of Boxes
* Functional Activities Questionnaire

Age, sex, education, assessment timing, missingness, and module availability are also represented.

The final hybrid system contains five fold-specific Snapshot pipelines and five fold-specific Transformer checkpoints. Only one route is used for each patient history, after which fold-level hazards are averaged and converted to monotonic cumulative risks.

---

## Results

### ADNI development evaluation

**1,425 participants · 4,223 dynamic MCI landmarks**

| Horizon | AUROC | IPCW AUPRC | IPCW Brier |
| ------- | ----: | ---------: | ---------: |
| 1 year  | 0.815 |      0.332 |     0.0869 |
| 2 years | 0.844 |      0.641 |     0.1366 |
| 3 years | 0.861 |      0.761 |     0.1463 |
| 5 years | 0.887 |      0.877 |     0.1378 |

### NACC frozen external evaluation

**12,052 participants · 26,303 dynamic MCI landmarks**

The exact ADNI-trained system was evaluated without retraining or initial external recalibration. ADAS13 was treated as structurally unavailable.

| Horizon | IPCW AUROC (95% CI) | IPCW AUPRC | IPCW Brier |
| ------- | ------------------: | ---------: | ---------: |
| 1 year  | 0.719 (0.703–0.735) |      0.123 |     0.0556 |
| 2 years | 0.733 (0.724–0.743) |      0.441 |     0.1756 |
| 3 years | 0.759 (0.750–0.768) |      0.619 |     0.2042 |
| 5 years | 0.778 (0.768–0.789) |      0.762 |     0.2024 |

External discrimination transported better than absolute-risk calibration. Secondary cross-fitted recalibration improved calibration without retraining the underlying prediction models.

---

## Try the demo

The hosted Hugging Face Space provides model-backed inference using a separately deployed frozen artifact bundle:

**[Launch the interactive research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)**

The public interface accepts demographic, cognitive, functional, and assessment-date inputs and automatically selects the Snapshot or longitudinal route.

### Local source-level check

```bash
git clone https://github.com/rey-Lii/mci-ad-risk-assessment.git
cd mci-ad-risk-assessment
pip install -e ".[test]"
pytest
python examples/quickstart.py
```

The local quickstart demonstrates validation, routing, temporal tensor construction, and module-availability handling. Fitted model weights and preprocessors are not redistributed in this compact repository.

---

## Repository structure

```text
src/ra_fmlr/
├── data.py          input validation and tensor construction
├── model.py         modular Transformer and route selection
├── inference.py     frozen hybrid inference contract
├── evaluation.py    hazard-to-risk and metric helpers
└── training.py      wrappers for the complete frozen pipeline

examples/            synthetic quickstart
tests/               validation, model, and risk tests
docs/                model card and supporting documentation
reports/public/      aggregate public evaluation outputs
```

Raw ADNI/NACC data, patient-level predictions, fitted preprocessors, and trained checkpoints are not redistributed.

---


## Contact

**Qirui Li**
GitHub: [rey-Lii](https://github.com/rey-Lii)
Email: [liqirui019@gmail.com](mailto:liqirui019@gmail.com)
