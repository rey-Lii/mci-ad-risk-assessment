"""Frozen V6.1-Hybrid-QC patient-level research inference.

This module is the single canonical inference path for the final frozen system:

- one distinct assessment date:
    five-fold Step 18N resource-aware Snapshot logistic ensemble;
- two or more distinct assessment dates:
    five-fold post-CDRSB-QC V6.1 gap-controlled, latest-anchored
    modular temporal Transformer ensemble.

The module performs no training, recalibration, probability clipping, or
checkpoint modification. It averages conditional hazards across folds and then
derives cumulative 1-, 2-, 3-, and 5-year risks.

Research use only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
import torch

from .data import (
    CONTEXT_FEATURES,
    HORIZONS,
    MODULES,
    TOKEN_FEATURES,
    build_patient_tensor_package,
    normalize_visits,
    transform_fold_features,
)
from .model import (
    GapControlledLatestAnchoredModularTemporalTransformer,
    ModelConfig,
)


INTERVALS: tuple[int, ...] = (0, 1, 2, 3)
GAP_CHANNEL_INDEX = 4
SCENARIOS: tuple[str, ...] = (
    "natural",
    "no_ADAS13",
    "MMSE_plus_CDGLOBAL",
    "MMSE_only",
)
SCENARIO_ALLOWED_MODULES: dict[str, tuple[str, ...]] = {
    "natural": MODULES,
    "no_ADAS13": ("MMSE", "CDGLOBAL", "CDRSB", "FAQTOTAL"),
    "MMSE_plus_CDGLOBAL": ("MMSE", "CDGLOBAL"),
    "MMSE_only": ("MMSE",),
}

SNAPSHOT_RESULT_DIR = Path(
    "results/v61_history1_resource_aware_logistic_cdrsb_qc/fivefold_post_qc"
)
TRANSFORMER_PREPROCESSING_DIR = Path(
    "data/processed/adni_dynamic_v6_cdrsb_qc/preprocessing"
)
TRANSFORMER_RESULT_DIR = Path(
    "results/v61_gap_controlled_latest_anchored_transformer_cdrsb_qc/"
    "primary_candidate_post_qc"
)


@dataclass(frozen=True)
class FrozenSnapshotEnsemble:
    """Five fold-specific Snapshot preprocessors and interval models."""

    preprocessors: tuple[dict[str, Any], ...]
    models: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class FrozenLongitudinalEnsemble:
    """Five post-QC V6.1 Transformer folds and matching preprocessors."""

    preprocessors: tuple[dict[str, Any], ...]
    models: tuple[
        GapControlledLatestAnchoredModularTemporalTransformer, ...
    ]
    device: torch.device


@dataclass(frozen=True)
class FrozenHybridEnsemble:
    """The two frozen expert branches under one history router."""

    snapshot: FrozenSnapshotEnsemble
    longitudinal: FrozenLongitudinalEnsemble
    model_root: Path


@dataclass(frozen=True)
class HybridPredictionResult:
    """Unified routed inference output."""

    route: str
    route_label: str
    scenario: str
    history_depth: int
    history_span_days: int
    repeated_modules: tuple[str, ...]
    available_modules: tuple[str, ...]
    hazards: np.ndarray
    risks: np.ndarray
    fold_hazards: np.ndarray
    fold_risks: np.ndarray
    fold_risk_sd: np.ndarray
    fold_risk_min: np.ndarray
    fold_risk_max: np.ndarray
    agreement_label: str
    agreement_max_sd: float
    agreement_max_range: float


def hazards_to_risks(hazards: np.ndarray) -> np.ndarray:
    """Convert conditional interval hazards to cumulative risks."""
    array = np.asarray(hazards, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(INTERVALS):
        raise ValueError(
            "Hazards must have shape [n, 4] for intervals "
            "0-1, 1-2, 2-3, and 3-5 years."
        )
    if not np.isfinite(array).all():
        raise ValueError("Non-finite hazards.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("Hazards must be probabilities between 0 and 1.")
    return 1.0 - np.cumprod(1.0 - array, axis=1)


def resolve_frozen_hybrid_artifacts(
    model_root: Path,
) -> dict[str, tuple[Path, ...] | tuple[tuple[Path, ...], ...]]:
    """Resolve every artifact used by the final frozen Hybrid."""
    root = model_root.expanduser().resolve()

    snapshot_base = root / SNAPSHOT_RESULT_DIR
    snapshot_preprocessors = tuple(
        snapshot_base / "preprocessing" / f"fold_{fold}_preprocessor.json"
        for fold in range(5)
    )
    snapshot_models = tuple(
        tuple(
            snapshot_base
            / "models"
            / f"fold_{fold}_interval_{interval}.joblib"
            for interval in INTERVALS
        )
        for fold in range(5)
    )

    transformer_preprocessors = tuple(
        root
        / TRANSFORMER_PREPROCESSING_DIR
        / f"fold_{fold}_preprocessor.json"
        for fold in range(5)
    )
    transformer_checkpoints = tuple(
        root
        / TRANSFORMER_RESULT_DIR
        / "checkpoints"
        / f"fold_{fold}_best.pt"
        for fold in range(5)
    )

    return {
        "snapshot_preprocessors": snapshot_preprocessors,
        "snapshot_models": snapshot_models,
        "transformer_preprocessors": transformer_preprocessors,
        "transformer_checkpoints": transformer_checkpoints,
    }


def frozen_hybrid_artifact_status(
    model_root: Path | None,
) -> tuple[bool, list[str]]:
    """Return whether all frozen expert artifacts exist."""
    if model_root is None:
        return False, ["MCI_AD_MODEL_ROOT is not configured."]

    resolved = resolve_frozen_hybrid_artifacts(model_root)
    paths: list[Path] = []
    for value in resolved.values():
        if value and isinstance(value[0], tuple):
            for nested in value:
                paths.extend(nested)
        else:
            paths.extend(value)  # type: ignore[arg-type]

    missing = [str(path) for path in paths if not path.exists()]
    return len(missing) == 0, missing


def _safe_torch_load(
    path: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location=device)

    if not isinstance(payload, Mapping):
        raise TypeError(f"Unexpected checkpoint payload: {path}")
    return payload


def _unwrap_snapshot_model(payload: object, path: Path) -> object:
    """Extract exactly one predict_proba estimator from a joblib payload."""
    if hasattr(payload, "predict_proba"):
        return payload

    matches: dict[int, object] = {}
    visited: set[int] = set()
    preferred = (
        "model",
        "estimator",
        "classifier",
        "logistic_model",
        "interval_model",
        "fitted_model",
        "pipeline",
        "predictor",
    )

    def walk(value: object) -> None:
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)

        if hasattr(value, "predict_proba"):
            matches[object_id] = value
            return

        if isinstance(value, Mapping):
            ordered_keys = [key for key in preferred if key in value]
            ordered_keys.extend(key for key in value if key not in preferred)
            for key in ordered_keys:
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    if len(matches) != 1:
        raise TypeError(
            f"Expected exactly one predict_proba estimator in {path}; "
            f"found {len(matches)}."
        )
    return next(iter(matches.values()))


def load_frozen_hybrid_ensemble(
    model_root: Path,
    *,
    device_name: str = "cpu",
) -> FrozenHybridEnsemble:
    """Load both final expert ensembles without changing their parameters."""
    root = model_root.expanduser().resolve()
    resolved = resolve_frozen_hybrid_artifacts(root)

    status, missing = frozen_hybrid_artifact_status(root)
    if not status:
        raise FileNotFoundError(
            "Frozen Hybrid artifacts are missing:\n" + "\n".join(missing)
        )

    snapshot_preprocessors = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in resolved["snapshot_preprocessors"]  # type: ignore[index]
    )

    snapshot_models: list[tuple[object, ...]] = []
    for fold_paths in resolved["snapshot_models"]:  # type: ignore[index]
        fold_models = tuple(
            _unwrap_snapshot_model(joblib.load(path), path)
            for path in fold_paths
        )
        snapshot_models.append(fold_models)

    device = torch.device(device_name)
    transformer_preprocessors: list[dict[str, Any]] = []
    transformer_models: list[
        GapControlledLatestAnchoredModularTemporalTransformer
    ] = []

    for fold, (preprocessor_path, checkpoint_path) in enumerate(
        zip(
            resolved["transformer_preprocessors"],  # type: ignore[index]
            resolved["transformer_checkpoints"],  # type: ignore[index]
            strict=True,
        )
    ):
        preprocessor = json.loads(
            preprocessor_path.read_text(encoding="utf-8")
        )
        checkpoint = _safe_torch_load(checkpoint_path, device)

        observed_fold = int(checkpoint.get("outer_fold", -1))
        if observed_fold != fold:
            raise ValueError(
                f"Transformer checkpoint fold mismatch at {checkpoint_path}: "
                f"observed {observed_fold}, expected {fold}."
            )

        config_payload = checkpoint.get("model_config")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(config_payload, Mapping):
            raise ValueError(
                f"Checkpoint has no mapping model_config: {checkpoint_path}"
            )
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                f"Checkpoint has no mapping state_dict: {checkpoint_path}"
            )

        model = GapControlledLatestAnchoredModularTemporalTransformer(
            ModelConfig(**dict(config_payload))
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        transformer_preprocessors.append(preprocessor)
        transformer_models.append(model)

    return FrozenHybridEnsemble(
        snapshot=FrozenSnapshotEnsemble(
            preprocessors=snapshot_preprocessors,
            models=tuple(snapshot_models),
        ),
        longitudinal=FrozenLongitudinalEnsemble(
            preprocessors=tuple(transformer_preprocessors),
            models=tuple(transformer_models),
            device=device,
        ),
        model_root=root,
    )


def _validate_and_summarize_visits(
    visits: Iterable[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    int,
    int,
    tuple[str, ...],
    tuple[str, ...],
]:
    normalized = normalize_visits(visits)

    empty_visits = [
        index
        for index, visit in enumerate(normalized, start=1)
        if not any(visit[module] is not None for module in MODULES)
    ]
    if empty_visits:
        listing = ", ".join(str(index) for index in empty_visits)
        raise ValueError(
            "Every assessment date must contain at least one score. "
            f"Empty assessment(s): {listing}."
        )

    first_date = normalized[0]["date"]
    latest_date = normalized[-1]["date"]
    history_span_days = int((latest_date - first_date).days)

    repeated_modules = tuple(
        module
        for module in MODULES
        if sum(visit[module] is not None for visit in normalized) >= 2
    )
    available_modules = tuple(
        module
        for module in MODULES
        if any(visit[module] is not None for visit in normalized)
    )

    return (
        normalized,
        len(normalized),
        history_span_days,
        repeated_modules,
        available_modules,
    )


def _build_snapshot_frame(
    *,
    age: float,
    sex_male: int,
    education_years: float,
    visit: Mapping[str, object],
) -> pd.DataFrame:
    row: dict[str, float | int] = {
        "age_at_landmark": float(age),
        "sex_male": int(sex_male),
        "education_years": float(education_years),
    }

    for module in MODULES:
        value = visit.get(module)
        available = value is not None
        row[f"{module}_available"] = int(available)
        row[f"{module}_latest_value"] = (
            float(value) if available else np.nan
        )

    return pd.DataFrame([row])


def _snapshot_effective_availability(
    frame: pd.DataFrame,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown resource scenario: {scenario}")

    natural = np.column_stack(
        [
            pd.to_numeric(
                frame[f"{module}_available"],
                errors="coerce",
            )
            .fillna(0)
            .eq(1)
            .to_numpy(dtype=bool)
            for module in MODULES
        ]
    )
    allowed = set(SCENARIO_ALLOWED_MODULES[scenario])
    structural = np.asarray(
        [module in allowed for module in MODULES],
        dtype=bool,
    )
    effective = natural & structural[None, :]
    return natural, effective


def _transform_snapshot_frame(
    frame: pd.DataFrame,
    preprocessor: Mapping[str, Any],
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact final Step 18N fold-specific Snapshot transformation."""
    n_rows = len(frame)
    columns: list[np.ndarray] = []

    context_stats = preprocessor["context_stats"]
    for feature in CONTEXT_FEATURES:
        spec = context_stats[feature]
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        missing = ~np.isfinite(values)
        values[missing] = float(spec["median"])
        if bool(spec["standardize"]):
            std = max(float(spec["std"]), 1e-8)
            values = (values - float(spec["mean"])) / std
        columns.extend([values, missing.astype(np.float64)])

    natural_available, effective_available = (
        _snapshot_effective_availability(frame, scenario)
    )

    module_stats = preprocessor["module_stats"]
    for module_index, module in enumerate(MODULES):
        spec = module_stats[module]
        values = pd.to_numeric(
            frame[f"{module}_latest_value"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        finite_observed = (
            np.isfinite(values) & natural_available[:, module_index]
        )
        standardized = np.zeros(n_rows, dtype=np.float64)
        std = max(float(spec["std"]), 1e-8)
        standardized[finite_observed] = (
            values[finite_observed] - float(spec["mean"])
        ) / std
        standardized[~effective_available[:, module_index]] = 0.0

        columns.extend(
            [
                standardized,
                effective_available[:, module_index].astype(np.float64),
            ]
        )

    matrix = np.column_stack(columns).astype(np.float64)
    if matrix.shape[1] != 16:
        raise RuntimeError(
            f"Snapshot feature width mismatch: {matrix.shape[1]} != 16."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite values remain after Snapshot preprocessing.")

    runnable = effective_available.sum(axis=1) > 0
    return matrix, runnable


def _predict_positive(model: object, matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(
        model.predict_proba(matrix),
        dtype=np.float64,
    )
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Unexpected Snapshot predict_proba output shape: "
            f"{probabilities.shape}."
        )
    positive = probabilities[:, 1]
    if not np.isfinite(positive).all():
        raise ValueError("Non-finite Snapshot probabilities.")
    return positive


def _predict_snapshot_folds(
    frame: pd.DataFrame,
    ensemble: FrozenSnapshotEnsemble,
    scenario: str,
) -> np.ndarray:
    fold_hazards: list[np.ndarray] = []

    for preprocessor, interval_models in zip(
        ensemble.preprocessors,
        ensemble.models,
        strict=True,
    ):
        matrix, runnable = _transform_snapshot_frame(
            frame,
            preprocessor,
            scenario,
        )
        if not bool(runnable[0]):
            raise ValueError(
                f"The selected assessment is not runnable under {scenario}."
            )

        hazards = np.column_stack(
            [
                _predict_positive(model, matrix)
                for model in interval_models
            ]
        )
        fold_hazards.append(hazards[0])

    return np.stack(fold_hazards, axis=0)


def _apply_longitudinal_scenario(
    package: dict[str, np.ndarray],
    scenario: str,
) -> dict[str, np.ndarray]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown resource scenario: {scenario}")

    allowed = set(SCENARIO_ALLOWED_MODULES[scenario])
    output = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in package.items()
    }

    for module_index, module in enumerate(MODULES):
        if module in allowed:
            continue
        output["token_features_raw"][:, module_index, :, :] = 0.0
        output["observation_mask"][:, module_index, :] = 0
        output["module_available"][:, module_index] = 0
        output["module_lengths"][:, module_index] = 0

    if not output["module_available"].astype(bool).any():
        raise ValueError(
            f"The selected assessment is not runnable under {scenario}."
        )
    return output


def _transform_longitudinal_fold(
    package: dict[str, np.ndarray],
    preprocessor: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact post-QC V6.1 transform: Stage 18 transform plus zeroed gap."""
    token, context = transform_fold_features(package, preprocessor)
    token = token.copy()
    token[..., GAP_CHANNEL_INDEX] = 0.0
    return token.astype(np.float32), context.astype(np.float32)


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, Mapping):
        for key in ("logits", "hazard_logits", "interval_logits"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]

    raise TypeError(
        "Could not extract interval logits from Transformer output of type "
        f"{type(output).__name__}."
    )


def _predict_longitudinal_folds(
    package: dict[str, np.ndarray],
    ensemble: FrozenLongitudinalEnsemble,
    scenario: str,
) -> np.ndarray:
    scenario_package = _apply_longitudinal_scenario(package, scenario)

    observation_mask = scenario_package["observation_mask"].astype(bool)
    module_available = scenario_package["module_available"].astype(bool)
    fold_hazards: list[np.ndarray] = []

    for model, preprocessor in zip(
        ensemble.models,
        ensemble.preprocessors,
        strict=True,
    ):
        token, context = _transform_longitudinal_fold(
            scenario_package,
            preprocessor,
        )
        with torch.inference_mode():
            output = model(
                torch.from_numpy(token).to(ensemble.device),
                torch.from_numpy(observation_mask).to(ensemble.device),
                torch.from_numpy(module_available).to(ensemble.device),
                torch.from_numpy(context).to(ensemble.device),
            )
            logits = _extract_logits(output)
            hazards = torch.sigmoid(logits).detach().cpu().numpy()

        if hazards.shape != (1, len(INTERVALS)):
            raise ValueError(
                f"Unexpected Transformer hazard shape: {hazards.shape}."
            )
        fold_hazards.append(hazards[0].astype(np.float64))

    return np.stack(fold_hazards, axis=0)


def _agreement_label(
    fold_risks: np.ndarray,
) -> tuple[str, float, float]:
    risk_sd = fold_risks.std(axis=0, ddof=0)
    risk_range = fold_risks.max(axis=0) - fold_risks.min(axis=0)
    max_sd = float(risk_sd.max())
    max_range = float(risk_range.max())

    # Presentation-oriented ensemble consistency bands. These describe
    # fold agreement only; they are not clinical confidence intervals.
    if max_sd <= 0.05 and max_range <= 0.15:
        label = "High"
    elif max_sd <= 0.10 and max_range <= 0.30:
        label = "Moderate"
    else:
        label = "Variable"
    return label, max_sd, max_range


def _assemble_result(
    *,
    route: str,
    route_label: str,
    scenario: str,
    history_depth: int,
    history_span_days: int,
    repeated_modules: tuple[str, ...],
    available_modules: tuple[str, ...],
    fold_hazards: np.ndarray,
) -> HybridPredictionResult:
    if fold_hazards.shape != (5, len(INTERVALS)):
        raise ValueError(
            f"Expected fold hazards [5, 4], got {fold_hazards.shape}."
        )

    fold_risks = hazards_to_risks(fold_hazards)
    ensemble_hazards = fold_hazards.mean(axis=0)
    ensemble_risks = hazards_to_risks(
        ensemble_hazards.reshape(1, -1)
    )[0]
    agreement, max_sd, max_range = _agreement_label(fold_risks)

    return HybridPredictionResult(
        route=route,
        route_label=route_label,
        scenario=scenario,
        history_depth=history_depth,
        history_span_days=history_span_days,
        repeated_modules=repeated_modules,
        available_modules=available_modules,
        hazards=ensemble_hazards,
        risks=ensemble_risks,
        fold_hazards=fold_hazards,
        fold_risks=fold_risks,
        fold_risk_sd=fold_risks.std(axis=0, ddof=0),
        fold_risk_min=fold_risks.min(axis=0),
        fold_risk_max=fold_risks.max(axis=0),
        agreement_label=agreement,
        agreement_max_sd=max_sd,
        agreement_max_range=max_range,
    )


def predict_frozen_hybrid(
    *,
    age: float,
    sex_male: int,
    education_years: float,
    visits: Iterable[dict[str, object]],
    ensemble: FrozenHybridEnsemble,
    scenario: str = "natural",
) -> HybridPredictionResult:
    """Route by distinct assessment-date depth and predict with frozen folds."""
    if scenario not in SCENARIOS:
        raise ValueError(
            f"Scenario must be one of: {', '.join(SCENARIOS)}."
        )

    (
        normalized,
        history_depth,
        history_span_days,
        repeated_modules,
        available_modules,
    ) = _validate_and_summarize_visits(visits)

    if history_depth == 1:
        frame = _build_snapshot_frame(
            age=age,
            sex_male=sex_male,
            education_years=education_years,
            visit=normalized[0],
        )
        fold_hazards = _predict_snapshot_folds(
            frame,
            ensemble.snapshot,
            scenario,
        )
        return _assemble_result(
            route="single_visit",
            route_label="Snapshot survival expert",
            scenario=scenario,
            history_depth=history_depth,
            history_span_days=history_span_days,
            repeated_modules=repeated_modules,
            available_modules=available_modules,
            fold_hazards=fold_hazards,
        )

    package = build_patient_tensor_package(
        age=age,
        sex_male=sex_male,
        education_years=education_years,
        visits=normalized,
    )
    fold_hazards = _predict_longitudinal_folds(
        package,
        ensemble.longitudinal,
        scenario,
    )
    return _assemble_result(
        route="longitudinal_v61",
        route_label="Modular longitudinal Transformer",
        scenario=scenario,
        history_depth=history_depth,
        history_span_days=history_span_days,
        repeated_modules=repeated_modules,
        available_modules=available_modules,
        fold_hazards=fold_hazards,
    )



def route_patient_history(history_depth: int) -> str:
    from .model import select_route
    return select_route(history_depth)
