# Secondary NACC Recalibration Analysis

This directory contains aggregate outputs from a secondary local model-updating analysis of the frozen V6.1-Hybrid-QC NACC predictions.

- Recalibration used patient-level fivefold cross-fitting.
- Snapshot and Longitudinal Transformer routes were updated separately.
- Calibration was performed on the four discrete-time hazard intervals.
- The frozen zero-shot NACC evaluation remains the primary external validation result.
- Recalibrated results must not be interpreted as zero-shot external performance.

## Public contents

The released files contain aggregate metrics, calibration coefficients, fold-level counts, calibration curves, and audit checks only.

Participant-level data, identifiers, raw predictions, recalibrated patient-level predictions, fitted model weights, and fold-specific preprocessors are not distributed.

## Main aggregate results

| Analysis | Mean horizon Brier | Integrated Brier, 0–5 years | Mean absolute calibration gap |
|---|---:|---:|---:|
| Frozen zero-shot | 0.1594 | 0.1480 | 0.0666 |
| Cross-fitted intercept-only | 0.1513 | 0.1399 | 0.0268 |
| Cross-fitted intercept and slope | 0.1494 | 0.1383 | 0.0340 |
