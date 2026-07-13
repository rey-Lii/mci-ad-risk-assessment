# History- and Resource-Adaptive MCI-to-AD Risk Modeling

[![tests](https://github.com/rey-Lii/mci-ad-longitudinal-risk-prediction2/actions/workflows/tests.yml/badge.svg)](https://github.com/rey-Lii/mci-ad-longitudinal-risk-prediction2/actions/workflows/tests.yml)

A frozen retrospective research prototype for dynamic **1-, 2-, 3-, and 5-year risk assessment of progression from mild cognitive impairment (MCI) to Alzheimer’s disease dementia**.

**V6.1-Hybrid-QC** combines a regularized Snapshot survival expert with a latest-anchored modular longitudinal Transformer. The prediction route adapts to the amount of available patient history, while the modular representation supports heterogeneous assessment availability.

> **Research use only.** This repository is not a clinical device and is not intended for diagnosis, treatment selection, triage, or patient-level prognosis communication.

## Research question

Clinical histories differ in both depth and resource availability. Some patients have only one assessment; others have irregular longitudinal follow-up. Cognitive and functional modules may also be missing or structurally unavailable across cohorts.

This project asks:

> Can one risk system adapt to both history depth and assessment availability while retaining transportable discrimination in an independent cohort?

## Main contributions

1. **History-adaptive routing**
   A deterministic router assigns one-date histories to a Snapshot discrete-time survival expert and histories with two or more distinct dates to a longitudinal Transformer.

2. **Resource-adaptive longitudinal modeling**
   Five assessment modules are represented separately, with availability-aware fusion, irregular-time features, trajectory summaries, and latest-state anchoring.

3. **Frozen external evaluation**
   The exact ADNI-trained system was evaluated zero-shot in NACC without retraining or external recalibration, under structural unavailability of ADAS13.

4. **Auditable public package**
   This compact repository exposes input validation, tensor construction, model architecture, routing, risk conversion, synthetic examples, tests, and aggregate results while excluding restricted patient-level data and fitted artifacts.

## System overview

```text
Patient history
      |
      +-- one distinct assessment date
      |       -> Snapshot survival expert
      |
      +-- two or more distinct assessment dates
              -> latest-anchored modular Transformer
                         |
                         -> conditional hazards
                         -> cumulative risk at 1, 2, 3, and 5 years
```

### Inputs

- ADAS13
- MMSE
- global CDR
- CDR Sum of Boxes
- FAQ total score
- age, sex, and education
- assessment dates and irregular time gaps
- explicit module-availability information

Each assessment date must contain at least one observed cognitive or functional score. Demographic and score ranges are validated before tensor construction or routed inference.

## Evaluation summary

### ADNI development evaluation

The frozen development evaluation included **1,425 participants** and **4,223 dynamic MCI landmarks**. Natural-availability, patient-grouped out-of-fold discrimination was:

| Horizon | AUROC |
|---|---:|
| 1 year | 0.815 |
| 2 years | 0.844 |
| 3 years | 0.861 |
| 5 years | 0.887 |

### Independent NACC evaluation

The exact frozen system was evaluated without retraining or external recalibration in **12,052 participants** and **26,303 dynamic MCI landmarks**. ADAS13 was structurally unavailable.

| Horizon | IPCW AUROC (95% CI) | IPCW AUPRC | IPCW Brier |
|---|---:|---:|---:|
| 1 year | 0.719 (0.703–0.735) | 0.123 | 0.056 |
| 2 years | 0.733 (0.724–0.743) | 0.441 | 0.176 |
| 3 years | 0.759 (0.750–0.768) | 0.619 | 0.204 |
| 5 years | 0.778 (0.768–0.789) | 0.762 | 0.202 |

External discrimination transported better than absolute-risk calibration. Target-population recalibration and prospective evaluation would be required before any clinical use.

Full evaluation details and limitations are documented in the [Model Card](MODEL_CARD.md).

## My contribution

**Qirui Li** designed and implemented the research workflow represented by this release, including longitudinal clinical data engineering, dynamic landmark construction, hybrid history routing, modular temporal modeling, internal evaluation, independent NACC validation, quality-control audits, and public research-demo packaging.

The project was developed as an independent medical-AI research prototype focused on longitudinal disease modeling, missing assessment modules, external transportability, and clinically realistic data constraints.

## Quick check

Python 3.10 or later is required.

```bash
pip install -e ".[test]"
pytest
python examples/quickstart.py
```

The synthetic quickstart demonstrates:

- validated patient input;
- automatic Snapshot versus longitudinal routing;
- V6 tensor construction;
- module-availability representation.

It intentionally does not return hard-coded risks. Real prediction requires private frozen fold-specific weights and fitted preprocessors.

## Repository map

```text
src/ra_fmlr/
├── data.py          # validation, date normalization, and tensor construction
├── model.py         # modular longitudinal Transformer and router
├── inference.py     # frozen hybrid inference contract
├── evaluation.py    # hazard-to-risk and metric helpers
└── training.py      # wrappers for frozen numbered pipeline stages

examples/
└── quickstart.py    # synthetic routing and tensor example

tests/               # public contract, temporal-feature, model, and risk tests
results/             # copied aggregate frozen results
```

## Reproducibility boundary

Included publicly:

- source architecture and routing logic;
- patient-input validation and tensor construction;
- synthetic examples;
- automated tests for Python 3.10 and 3.11;
- aggregate evaluation results;
- model card and data-use boundaries.

Not distributed:

- participant-level ADNI or NACC data;
- patient identifiers;
- fitted model weights;
- fitted fold-specific preprocessors;
- patient-level predictions.

The compact repository passed a 13-check parity audit against the complete frozen release. The numbered cohort, quality-control, training, audit, and external-validation workflow remains available in the [complete frozen research pipeline](https://github.com/rey-Lii/mci-ad-resource-adaptive-transformerrr).

## Limitations

- retrospective development and evaluation in selected research cohorts;
- no prospective clinical validation;
- external calibration drift, especially at longer horizons;
- death was not modeled as a competing event;
- missingness may remain informative;
- repeated landmarks originate from the same participants;
- fitted weights and preprocessors are not publicly distributed.

## Links

- [Hosted research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)
- [Model Card](MODEL_CARD.md)
- [Complete frozen research and audit pipeline](https://github.com/rey-Lii/mci-ad-resource-adaptive-transformerrr)
