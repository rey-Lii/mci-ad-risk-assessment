# Data Access and Redistribution Boundary

## Source datasets

This project uses data from:

- the Alzheimer’s Disease Neuroimaging Initiative (ADNI);
- the National Alzheimer’s Coordinating Center (NACC).

Both datasets require approval and acceptance of their respective data-use
terms. This repository does not provide source data or participant-level
derivatives.

## Files that must remain local

Do not commit or upload:

- `data/data_raw/` or other raw-data directories;
- `data/interim/`;
- `data/processed/`;
- patient-level CSV or Parquet files;
- patient or landmark identifiers;
- longitudinal tensors (`.npz`, `.npy`);
- participant-level predictions or joined evaluation tables;
- fold manifests containing participant identifiers;
- serialized imputers, scalers, or preprocessing objects derived from
  restricted data;
- model checkpoints or fitted interval models unless separately reviewed for
  release;
- private inference backends, local archives, and backups.

## What may be shared

The public repository may contain:

- source code;
- configuration templates without local paths or credentials;
- tests;
- feature, routing, and outcome definitions;
- model architecture definitions;
- aggregate cohort counts;
- aggregate metrics and confidence intervals;
- calibration bins containing no individual records;
- aggregate audit summaries and manifests containing no participant
  identifiers;
- the frozen external-validation protocol and aggregate NACC results.

## Reproducing the analysis

1. Apply separately for ADNI and NACC access.
2. Download approved data directly from the providers.
3. Keep source data and participant-level derivatives outside Git tracking.
4. Use the [complete frozen research pipeline](https://github.com/rey-Lii/mci-ad-resource-adaptive-transformerrr) for cohort
   construction, model fitting, quality-control audits, and formal external
   validation.
5. Install this compact package for inspection of the final model, patient-input
   contract, routing behavior, evaluation helpers, and aggregate frozen results.
6. Supply authorized private fitted artifacts only when running frozen
   patient-level inference.

The compact repository is a public-facing, parity-checked presentation layer;
the complete frozen pipeline remains the authoritative implementation for full
end-to-end reproduction.

## Trained model weights

The public release excludes trained checkpoints and fitted preprocessors.
Release of artifacts trained on restricted participant-level data requires
separate review against provider data-use terms and disclosure risks.

## Research-use statement

The repository is for retrospective research and reproducibility. It is not a
medical device and is not intended for diagnosis, treatment, triage, or
patient-level prognosis communication.
