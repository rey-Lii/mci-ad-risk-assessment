# Low-Burden, History- and Availability-Aware MCI-to-AD Risk Prediction

[![Tests](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml/badge.svg)](https://github.com/rey-Lii/mci-ad-risk-assessment/actions/workflows/tests.yml)

**A history-adaptive and availability-aware clinical-AI system using routine cognitive and functional assessments, developed on ADNI and externally evaluated on NACC.**

[Open the model-backed research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)

> **Research use only.** Not for clinical use. Do not enter real patient information.

---

## Overview

This repository presents an externally evaluated pipeline for dynamic 1-, 2-, 3-, and 5-year prediction of progression from mild cognitive impairment (MCI) to Alzheimer’s disease dementia.

Clinical histories vary across healthcare settings. Some patients have only one assessment date, while others have repeated but irregular follow-up. Cognitive and functional assessments may also be unavailable across patients or cohorts, particularly in primary-care, community, and resource-limited settings.

Rather than forcing all patients through one sequence model, the system uses deterministic history-based routing:

- **one assessment date** → Snapshot survival expert;
- **repeated assessments** → modular longitudinal Transformer.

The longitudinal branch models five cognitive and functional assessment domains separately and explicitly represents module availability.

The system uses low-burden clinical information without requiring PET, CSF, MRI, or genetic testing.

---

## Highlights

- **Task:** Dynamic 1-, 2-, 3-, and 5-year MCI-to-AD risk prediction.
- **Clinical setting:** Low-burden risk stratification using routine clinical assessments.
- **History adaptation:** Different prediction pathways for single-date and longitudinal histories.
- **Availability-aware modeling:** Explicit representation of missing or unavailable assessment modules.
- **External evaluation:** Frozen ADNI-trained system evaluated on NACC without retraining.
- **Transparent workflow:** Public inference pipeline, validation tests, and aggregate evaluation reports.

---

## System overview

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/7aef8a20-1c11-4ba9-a8dd-522bbde6c469"
    alt="Overview of the history- and resource-adaptive hybrid survival system"
    width="82%"
  />
</p>

The system combines two prediction experts:

- a regularized Snapshot survival expert for patients with a single assessment date;
- a modular longitudinal Transformer for patients with repeated assessments.

Both branches estimate discrete-time hazards and convert them into cumulative 1-, 2-, 3-, and 5-year risks.

---

## Results at a glance

| Evaluation | Cohort | Participants | Dynamic landmarks | AUROC range |
|---|---|---:|---:|---:|
| Internal development evaluation | ADNI | 1,425 | 4,223 | 0.815–0.887 |
| Frozen external evaluation | NACC | 12,052 | 26,303 | 0.719–0.778 |

The frozen ADNI-trained system was evaluated on NACC without model retraining, feature revision, hyperparameter reselection, or initial external recalibration.

The system retained moderate external discrimination, while calibration differences between cohorts were assessed separately.

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

Only information available at or before each landmark is used for prediction, and patient-level grouping is maintained during evaluation to reduce information leakage.

### History-adaptive routing

The routing rule is deterministic:

```text
one assessment date
        ↓
Snapshot survival expert

two or more assessment dates
        ↓
Modular longitudinal Transformer
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

---

## Local validation

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

## Repository structure

```text
src/ra_fmlr/
├── data.py          input validation and tensor construction
├── model.py         modular Transformer and route selection
├── inference.py     frozen artifact loading and prediction pipeline
├── evaluation.py    risk conversion and metric utilities
└── training.py      training wrappers

examples/
tests/
docs/
reports/public/
```

---

## Contact

**Qirui Li**  
GitHub: [rey-Lii](https://github.com/rey-Lii)  
Email: liqirui019@gmail.com
