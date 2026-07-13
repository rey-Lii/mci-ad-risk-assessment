# Model Card

## Model identity

- **Version:** V6.1-Hybrid-QC
- **Task:** dynamic prediction of progression from MCI to Alzheimer’s disease
  dementia
- **Horizons:** 1, 2, 3, and 5 years
- **Status:** frozen retrospective research prototype

## Intended use

The model is intended for retrospective research on longitudinal disease
progression, heterogeneous assessment availability, dynamic survival
prediction, model robustness, external transportability, and calibration.

## Out-of-scope use

The system is not intended for diagnosis, treatment selection, triage,
patient-level prognosis communication, autonomous decision-making, or
uncalibrated clinical deployment.

## Intended population

Eligible prediction landmarks represent people assessed as having MCI, with no
prior dementia before the landmark. Predictions outside that population are
out of the validated task domain.

## Inputs

The input contract includes irregular histories from:

- ADAS13
- MMSE
- global CDR
- CDR Sum of Boxes
- FAQ total score
- age, sex, and education

Resource scenarios allow clinical modules to be unavailable. In the independent
NACC validation, ADAS13 was structurally unavailable.

## Architecture

A deterministic history-depth router assigns:

- one distinct assessment date to a regularized Snapshot survival expert;
- two or more distinct assessment dates to a modular longitudinal Transformer.

Both branches output discrete-time conditional hazards and cumulative risk at
1, 2, 3, and 5 years.

## Development evaluation

The frozen ADNI development evaluation included 1,425 patients and 4,223
dynamic landmarks. Natural-availability patient-grouped out-of-fold AUROCs were:

| Horizon | AUROC |
|---|---:|
| 1 year | 0.815 |
| 2 years | 0.844 |
| 3 years | 0.861 |
| 5 years | 0.887 |

## Independent external evaluation

The exact frozen model was evaluated zero-shot in NACC without retraining or
external recalibration:

- 12,052 patients
- 26,303 dynamic MCI landmarks
- structurally unavailable ADAS13
- participant-balanced IPCW metrics
- 1,000 patient-level bootstrap replicates

| Horizon | IPCW AUROC (95% CI) | IPCW AUPRC | IPCW Brier |
|---|---:|---:|---:|
| 1 year | 0.719 (0.703–0.735) | 0.123 | 0.056 |
| 2 years | 0.733 (0.724–0.743) | 0.441 | 0.176 |
| 3 years | 0.759 (0.750–0.768) | 0.619 | 0.204 |
| 5 years | 0.778 (0.768–0.789) | 0.762 | 0.202 |

Route-stratified analyses were descriptive because Snapshot and longitudinal
landmarks differ in history depth and risk composition.

## Limitations

- retrospective development and external evaluation in selected research
  cohorts;
- no prospective clinical validation;
- external calibration drift, particularly for longer-horizon absolute risk;
- death was not modeled as a competing event;
- non-AD dementia was treated as a terminal competing endpoint in the primary
  cause-specific evaluation;
- missingness may be informative;
- repeated landmarks originate from the same patients;
- trained weights and fitted preprocessors are not distributed publicly.

External discrimination transported better than absolute-risk calibration.
Target-population calibration and prospective evaluation are required before
clinical use.
