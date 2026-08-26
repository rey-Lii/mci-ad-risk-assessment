# Dynamic MCI-to-AD Risk Prediction Using Routine Clinical Assessments

[![Tests](https://github.com/rey-Lii/dynamic-mci-ad-risk-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/rey-Lii/dynamic-mci-ad-risk-prediction/actions/workflows/tests.yml)

**A history-adaptive framework for dynamic MCI-to-AD risk prediction from routine clinical assessments, developed on ADNI and externally evaluated on NACC.**

[Open the model-backed research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)

> **Research use only.** Not for clinical use. Do not enter real patient information.

---

## Overview

This repository presents an externally evaluated framework for dynamic 1-, 2-, 3-, and 5-year prediction of progression from mild cognitive impairment (MCI) to Alzheimer’s disease dementia using routinely collected cognitive and functional assessments.

Clinical histories vary in both depth and completeness: some patients have only one usable assessment date, while others have repeated but irregular follow-up, and individual assessments may be missing or unavailable.

**We therefore treat the structure of available history as part of the prediction problem itself.** Rather than forcing every patient through the same longitudinal model, the framework uses deterministic history-based routing:

- **one distinct assessment date** → Snapshot branch using a regularized discrete-time survival model;
- **two or more distinct assessment dates** → longitudinal branch using a modular temporal Transformer.

The longitudinal branch models five cognitive and functional assessments as separate temporal modules, explicitly representing availability, missingness, timing, recency, and local trajectories.

The framework relies on routine clinical information without requiring PET, CSF, MRI, or genetic testing, making the design particularly relevant to primary-care, community, and resource-constrained settings.

<p align="center">
  <img
    src="results/figures/hybrid_system_overview.png"
    alt="Overview of the history-adaptive and availability-aware hybrid survival system"
    width="82%"
  />
</p>

---


## Results at a glance

| Evaluation | Cohort | Participants | Dynamic landmarks | AUROC range |
|---|---|---:|---:|---:|
| Internal development evaluation | ADNI | 1,425 | 4,223 | 0.815–0.887 |
| Zero-shot external evaluation | NACC | 12,052 | 26,303 | 0.719–0.778 |

The NACC evaluation included **8,817 Snapshot-route** and **17,486 longitudinal-route landmarks**. The ADNI-trained model was applied without retraining; because compatible ADAS13 measurements were unavailable in NACC, predictions used the remaining available assessment modules.

### Performance by prediction horizon

| Horizon | ADNI AUROC | ADNI AUPRC | ADNI Brier | NACC AUROC | NACC AUPRC | NACC Brier |
|---|---:|---:|---:|---:|---:|---:|
| 1 year | 0.815 | 0.332 | 0.0869 | 0.719 | 0.123 | 0.0556 |
| 2 years | 0.844 | 0.641 | 0.1366 | 0.733 | 0.441 | 0.1756 |
| 3 years | 0.861 | 0.761 | 0.1463 | 0.759 | 0.619 | 0.2042 |
| 5 years | 0.887 | 0.877 | 0.1378 | 0.778 | 0.762 | 0.2024 |

---

## Data and variables

The framework uses age, sex, education, and five routine cognitive and functional assessments: ADAS13, MMSE, global Clinical Dementia Rating, Clinical Dementia Rating Sum of Boxes, and Functional Activities Questionnaire.

For repeated histories, each assessment is represented as a separate temporal module. Longitudinal features include the observed value, change from the first observation, annualized slope, relative timing, inter-assessment gaps, and recency to the prediction landmark, together with explicit observation and availability indicators.

Each eligible MCI assessment serves as a dynamic prediction landmark, using only information available at or before that date. Raw ADNI and NACC data are not included in this repository; access requires approval from the respective data providers.

---

## Research demo

[Launch the model-backed research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment), which accepts demographic and assessment histories and automatically selects the appropriate prediction branch.

Trained model weights are not included in the public repository.

---

## Contact

**Qirui Li**  
GitHub: [rey-Lii](https://github.com/rey-Lii)  
Email: liqirui019@gmail.com
