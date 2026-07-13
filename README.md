# History- and Resource-Adaptive MCI-to-AD Risk Modeling

## Links

- [Hosted research demo](https://huggingface.co/spaces/reylii/MCI-to-Alzheimers-Dementia-Risk-Assessment)
- [Complete frozen research and audit pipeline](https://github.com/rey-Lii/mci-ad-resource-adaptive-transformerrr)

Compact presentation of **V6.1-Hybrid-QC**, a retrospective research prototype
for dynamic 1-, 2-, 3-, and 5-year Alzheimer dementia risk assessment among
people with mild cognitive impairment.

> Research use only. Not a clinical device.

## Core system

```text
one distinct assessment date  -> Snapshot survival expert
two or more assessment dates  -> latest-anchored modular Transformer
```

- ADNI development: 1,425 participants / 4,223 dynamic landmarks.
- NACC validation: 12,052 participants / 26,303 landmarks.
- Frozen NACC `no_ADAS13` evaluation; no retraining or recalibration.

## Code map

- `src/ra_fmlr/model.py`: final Transformer and deterministic router
- `src/ra_fmlr/data.py`: patient validation and tensor construction
- `src/ra_fmlr/training.py`: wrappers for final frozen training stages
- `src/ra_fmlr/evaluation.py`: risk conversion and metric helpers
- `src/ra_fmlr/inference.py`: canonical frozen Hybrid inference
- `examples/quickstart.py`: synthetic input and route demonstration

## Quick check

```bash
pip install -e ".[test]"
pytest
python examples/quickstart.py
```

Patient-level data, identifiers, fitted weights, preprocessors, and private
predictions are not included. The complete frozen release remains the
methodological archive until parity testing is complete.

## Provenance and parity

This compact repository was generated from the complete frozen V6.1 public
release. Its Transformer configuration, state-dict contract, deterministic
forward behavior, patient tensor construction, history router, hazard-to-risk
conversion, and copied aggregate result files passed a 13-check parity audit.

The complete numbered cohort, quality-control, model, audit, and NACC
validation pipeline remains the authoritative implementation for full
methodological review.
