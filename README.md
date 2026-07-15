# Dynamic MCI-to-AD Risk Prediction from Routine Clinical Assessments

[![Tests](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml/badge.svg)](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml)

**A hybrid clinical-AI system for single-date and longitudinal patient histories, developed on ADNI and externally evaluated on NACC.**

[Open the model-backed research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)

> **Research use only.** Not for clinical use. Do not enter real patient information.

---

## Overview

This repository presents an externally evaluated pipeline for dynamic 1-, 2-, 3-, and 5-year prediction of progression from mild cognitive impairment (MCI) to Alzheimer’s disease dementia.

Clinical histories vary across healthcare settings. Some patients have only one assessment date, while others have repeated but irregular follow-up. Cognitive and functional assessments may also be unavailable across patients or cohorts, particularly in primary-care, community, and resource-limited settings.

Rather than forcing all patients through one sequence model, the system uses deterministic history-based routing:

- **one distinct assessment date** → regularized Snapshot survival expert;
- **two or more distinct assessment dates** → modular longitudinal Transformer.

Both branches use explicit module-availability information, while the longitudinal branch models the five cognitive and functional assessments as separate temporal modules.

The system uses low-burden clinical information without requiring PET, CSF, MRI, or genetic testing.


<p align="center">
  <img
    src="results/figures/hybrid_system_overview.png"
    alt="Overview of the history-adaptive and availability-aware hybrid survival system"
    width="82%"
  />
</p>

---

## Highlights

- **Task:** Dynamic 1-, 2-, 3-, and 5-year MCI-to-AD risk prediction.
- **Clinical setting:** Low-burden risk stratification using routine clinical assessments.
- **History adaptation:** Different prediction pathways for single-date and longitudinal histories.
- **Availability-aware modeling:** Explicit representation of missing or unavailable assessment modules.
- **External evaluation:** Frozen ADNI-trained system evaluated on NACC without retraining.
- **Transparent workflow:** Public architecture, routing logic, validation tests, and aggregate evaluation reports.

---

## Results at a glance

| Evaluation | Cohort | Participants | Dynamic landmarks | AUROC range |
|---|---|---:|---:|---:|
| Internal development evaluation | ADNI | 1,425 | 4,223 | 0.815–0.887 |
| Frozen zero-shot external evaluation | NACC | 12,052 | 26,303 | 0.719–0.778 |

The NACC evaluation comprised **8,817 Snapshot-route landmarks** and **17,486 longitudinal-route landmarks**, with **ADAS13 treated as structurally unavailable**. IPCW was used to account for right censoring.
### Horizon-specific performance

| Horizon | ADNI AUROC | ADNI AUPRC | ADNI Brier | NACC AUROC | NACC AUPRC | NACC Brier |
|---|---:|---:|---:|---:|---:|---:|
| 1 year | 0.815 | 0.332 | 0.0869 | 0.719 | 0.123 | 0.0556 |
| 2 years | 0.844 | 0.641 | 0.1366 | 0.733 | 0.441 | 0.1756 |
| 3 years | 0.861 | 0.761 | 0.1463 | 0.759 | 0.619 | 0.2042 |
| 5 years | 0.887 | 0.877 | 0.1378 | 0.778 | 0.762 | 0.2024 |

The frozen ADNI-trained system was evaluated zero-shot on NACC without retraining, feature changes, hyperparameter reselection, or external recalibration. These remain the primary external-validation results.

In a separate cross-fitted recalibration analysis, the mean absolute calibration gap decreased from **0.0666** to **0.0268** with intercept-only recalibration and to **0.0340** with intercept-and-slope recalibration, while the prediction models remained frozen.

---

## Data and variables

Raw ADNI and NACC participant-level data are not redistributed.

The system uses routine clinical information:

### Demographic information

- age;
- sex;
- education.

### Cognitive and functional assessments

Five assessment modules are modeled separately:

- ADAS13;
- MMSE;
- global Clinical Dementia Rating;
- Clinical Dementia Rating Sum of Boxes;
- Functional Activities Questionnaire.

The longitudinal representation additionally includes:

- assessment timing;
- irregular follow-up intervals;
- recency information;
- module-level missingness and availability indicators;
- trajectory summaries.

---

## Method overview

### Dynamic landmark construction

Each eligible MCI assessment is treated as a prediction landmark.

Only information available at or before each landmark is used for prediction, and all landmarks from the same participant are kept within a single evaluation partition to prevent participant overlap across partitions.

### History-adaptive routing

The routing rule is deterministic:

```text
one distinct assessment date
        ↓
regularized Snapshot survival expert

two or more distinct assessment dates
        ↓
modular longitudinal Transformer
```

This avoids applying a sequence model to histories without meaningful longitudinal information.

### Modular longitudinal modeling

Each cognitive and functional assessment module is represented separately.

The model distinguishes between observed assessments, missing assessments, and unavailable modules, allowing it to handle heterogeneous assessment patterns across patients and cohorts.

---

## Research demo

The hosted Hugging Face Space provides model-backed inference:

[Launch the interactive demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)

The interface accepts demographic information, cognitive and functional assessments, and assessment dates. The system automatically selects the appropriate prediction pathway.

The hosted demo uses a separately deployed frozen artifact bundle. Fitted preprocessors and model weights are not redistributed in this compact public repository.

---

## Local source-level validation

```bash
git clone https://github.com/rey-Lii/mci-ad-risk-assessment.git
cd mci-ad-risk-assessment

pip install -e ".[test]"

pytest
python examples/quickstart.py
```

The public repository demonstrates:

- input validation;
- temporal tensor construction;
- history-based routing;
- module-availability handling;
- risk conversion.

---

## Contact

**Qirui Li**  
GitHub: [rey-Lii](https://github.com/rey-Lii)  
Email: liqirui019@gmail.com
