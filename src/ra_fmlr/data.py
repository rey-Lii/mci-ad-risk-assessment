"""Patient-entry validation and tensor construction.

No ADNI/NACC source data are read and no input is persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PatientVisit:
    date: date | datetime | str
    ADAS13: float | None = None
    MMSE: float | None = None
    CDGLOBAL: float | None = None
    CDRSB: float | None = None
    FAQTOTAL: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "date": self.date,
            "ADAS13": self.ADAS13,
            "MMSE": self.MMSE,
            "CDGLOBAL": self.CDGLOBAL,
            "CDRSB": self.CDRSB,
            "FAQTOTAL": self.FAQTOTAL,
        }


@dataclass(frozen=True)
class PatientHistory:
    age: float
    sex_male: int
    education_years: float
    visits: tuple[PatientVisit, ...]

    @property
    def history_depth(self) -> int:
        normalized = normalize_visits(
            [visit.as_dict() for visit in self.visits]
        )
        return len(normalized)

    def build_tensor_package(self) -> dict[str, np.ndarray]:
        return build_patient_tensor_package(
            age=self.age,
            sex_male=self.sex_male,
            education_years=self.education_years,
            visits=[visit.as_dict() for visit in self.visits],
        )


MODULES: tuple[str, ...] = (
    "ADAS13",
    "MMSE",
    "CDGLOBAL",
    "CDRSB",
    "FAQTOTAL",
)


TOKEN_FEATURES: tuple[str, ...] = (
    "raw_value",
    "delta_from_first",
    "slope_from_first_per_year",
    "relative_time_years",
    "gap_years",
    "recency_years",
    "slope_valid",
    "gap_valid",
)


CONTEXT_FEATURES: tuple[str, ...] = (
    "age_at_landmark",
    "sex_male",
    "education_years",
)


HORIZONS: tuple[int, ...] = (1, 2, 3, 5)


MAX_HISTORY_DATES = 16


MINIMUM_SLOPE_DAYS = 30


MODULE_LABELS: dict[str, str] = {
    "ADAS13": (
        "Alzheimer’s Disease Assessment Scale–Cognitive Subscale, "
        "13-item (ADAS-Cog 13)"
    ),
    "MMSE": "Mini-Mental State Examination (MMSE)",
    "CDGLOBAL": "Clinical Dementia Rating (CDR) Global Score",
    "CDRSB": "Clinical Dementia Rating–Sum of Boxes (CDR-SB)",
    "FAQTOTAL": (
        "Functional Activities Questionnaire (FAQ) Total Score"
    ),
}


MODULE_RANGES: dict[str, tuple[float, float]] = {
    "ADAS13": (0.0, 85.0),
    "MMSE": (0.0, 30.0),
    "CDGLOBAL": (0.0, 3.0),
    "CDRSB": (0.0, 18.0),
    "FAQTOTAL": (0.0, 30.0),
}


CDGLOBAL_ALLOWED: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0)


def _as_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    parsed = datetime.fromisoformat(str(value))
    return parsed


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Assessment values must be finite numbers.")
    return number


def _validate_score(module: str, value: float) -> None:
    low, high = MODULE_RANGES[module]
    if not low <= value <= high:
        raise ValueError(
            f"{MODULE_LABELS[module]} must be between {low:g} and {high:g}."
        )
    if module == "CDGLOBAL" and value not in CDGLOBAL_ALLOWED:
        allowed = ", ".join(f"{item:g}" for item in CDGLOBAL_ALLOWED)
        raise ValueError(f"Global CDR must be one of: {allowed}.")


def validate_patient_context(
    *,
    age: object,
    sex_male: object,
    education_years: object,
) -> tuple[float, int, float]:
    """Validate and normalize demographic context shared by both routes."""
    age_value = float(age)
    sex_value = float(sex_male)
    education_value = float(education_years)

    if not np.isfinite(age_value) or not 40.0 <= age_value <= 110.0:
        raise ValueError("Age must be between 40 and 110 years.")
    if not np.isfinite(sex_value) or sex_value not in (0.0, 1.0):
        raise ValueError("Sex must be coded as female or male.")
    if (
        not np.isfinite(education_value)
        or not 0.0 <= education_value <= 30.0
    ):
        raise ValueError("Education must be between 0 and 30 years.")

    return age_value, int(sex_value), education_value


def normalize_visits(
    visits: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Validate visits and return them in chronological order."""
    normalized: list[dict[str, object]] = []
    for visit_index, visit in enumerate(visits, start=1):
        if "date" not in visit:
            raise ValueError(f"Assessment {visit_index} has no date.")
        visit_date = _as_datetime(visit["date"])
        row: dict[str, object] = {"date": visit_date}
        for module in MODULES:
            value = _as_optional_float(visit.get(module))
            if value is not None:
                _validate_score(module, value)
            row[module] = value
        if not any(row[module] is not None for module in MODULES):
            raise ValueError(
                f"Assessment {visit_index} must contain at least one score."
            )
        normalized.append(row)

    if not normalized:
        raise ValueError("At least one assessment visit is required.")
    if len(normalized) > MAX_HISTORY_DATES:
        raise ValueError(
            f"At most {MAX_HISTORY_DATES} assessment visits can be retained."
        )

    normalized.sort(key=lambda row: row["date"])
    dates = [row["date"] for row in normalized]
    if len(set(dates)) != len(dates):
        raise ValueError("Each assessment visit must have a different date.")

    observed = sum(
        row[module] is not None
        for row in normalized
        for module in MODULES
    )
    if observed == 0:
        raise ValueError("Enter at least one assessment score.")
    return normalized


def build_patient_tensor_package(
    *,
    age: float,
    sex_male: int,
    education_years: float,
    visits: Iterable[dict[str, object]],
    max_history_dates: int = MAX_HISTORY_DATES,
    minimum_slope_days: int = MINIMUM_SLOPE_DAYS,
) -> dict[str, np.ndarray]:
    """Build one exact V6 raw tensor package from longitudinal patient entries."""
    age, sex_male, education_years = validate_patient_context(
        age=age,
        sex_male=sex_male,
        education_years=education_years,
    )

    normalized = normalize_visits(visits)
    if len(normalized) > max_history_dates:
        normalized = normalized[-max_history_dates:]

    landmark_date = normalized[-1]["date"]
    observations: list[dict[str, object]] = []
    for visit in normalized:
        visit_date = visit["date"]
        for module in MODULES:
            value = visit[module]
            if value is not None:
                observations.append(
                    {
                        "date": visit_date,
                        "module": module,
                        "raw_value": float(value),
                    }
                )

    first_global_date = min(row["date"] for row in observations)
    token = np.zeros(
        (1, len(MODULES), max_history_dates, len(TOKEN_FEATURES)),
        dtype=np.float32,
    )
    observation_mask = np.zeros(
        (1, len(MODULES), max_history_dates),
        dtype=np.uint8,
    )

    for module_index, module in enumerate(MODULES):
        module_rows = sorted(
            (row for row in observations if row["module"] == module),
            key=lambda row: row["date"],
        )
        if not module_rows:
            continue

        first_module_date = module_rows[0]["date"]
        first_module_value = float(module_rows[0]["raw_value"])
        previous_date: datetime | None = None

        for position, row in enumerate(module_rows):
            observation_date = row["date"]
            raw_value = float(row["raw_value"])
            elapsed_days = (observation_date - first_module_date).days
            delta = raw_value - first_module_value
            slope_valid = position > 0 and elapsed_days >= minimum_slope_days
            slope = (
                delta / elapsed_days * 365.25
                if slope_valid and elapsed_days > 0
                else 0.0
            )
            gap_valid = previous_date is not None
            gap_days = (
                (observation_date - previous_date).days
                if previous_date is not None
                else 0
            )
            relative_days = (observation_date - first_global_date).days
            recency_days = (landmark_date - observation_date).days

            token[0, module_index, position] = np.asarray(
                [
                    raw_value,
                    delta,
                    slope,
                    relative_days / 365.25,
                    gap_days / 365.25,
                    recency_days / 365.25,
                    float(slope_valid),
                    float(gap_valid),
                ],
                dtype=np.float32,
            )
            observation_mask[0, module_index, position] = 1
            previous_date = observation_date

    module_lengths = observation_mask.sum(axis=2).astype(np.int16)
    module_available = (module_lengths > 0).astype(np.uint8)
    context_raw = np.asarray(
        [[age, float(sex_male), education_years]],
        dtype=np.float32,
    )
    context_missing = np.zeros((1, len(CONTEXT_FEATURES)), dtype=np.uint8)

    return {
        "token_features_raw": token,
        "observation_mask": observation_mask,
        "module_available": module_available,
        "module_lengths": module_lengths,
        "context_raw": context_raw,
        "context_missing": context_missing,
        "landmark_date": np.asarray(
            [landmark_date.strftime("%Y-%m-%d")],
            dtype=object,
        ),
        "module_names": np.asarray(MODULES),
        "token_feature_names": np.asarray(TOKEN_FEATURES),
        "context_feature_names": np.asarray(CONTEXT_FEATURES),
        "horizon_years": np.asarray(HORIZONS, dtype=np.int8),
    }


def transform_fold_features(
    package: dict[str, np.ndarray],
    preprocessor: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact fold-specific Stage 18 preprocessing contract."""
    token = package["token_features_raw"].astype(np.float32, copy=True)
    observation_mask = package["observation_mask"].astype(bool, copy=False)

    for module_index, module_name in enumerate(MODULES):
        stats_by_feature = preprocessor["module_continuous_stats"][
            module_name
        ]
        module_observed = observation_mask[:, module_index, :]
        for feature_index, feature_name in enumerate(TOKEN_FEATURES[:6]):
            values = token[:, module_index, :, feature_index]
            stats = stats_by_feature[feature_name]
            clip_low = stats.get("clip_low")
            clip_high = stats.get("clip_high")
            if clip_low is not None:
                values = np.maximum(values, float(clip_low))
            if clip_high is not None:
                values = np.minimum(values, float(clip_high))
            values = (
                values - float(stats["mean"])
            ) / max(float(stats["std"]), 1e-8)
            token[:, module_index, :, feature_index] = values

        slope_valid = token[:, module_index, :, 6] > 0.5
        gap_valid = token[:, module_index, :, 7] > 0.5
        token[:, module_index, :, 2] = np.where(
            slope_valid,
            token[:, module_index, :, 2],
            0.0,
        )
        token[:, module_index, :, 4] = np.where(
            gap_valid,
            token[:, module_index, :, 4],
            0.0,
        )
        token[:, module_index] = np.where(
            module_observed[..., None],
            token[:, module_index],
            0.0,
        )

    context = package["context_raw"].astype(np.float32, copy=True)
    context_missing = package["context_missing"].astype(bool)
    for feature_index, feature_name in enumerate(CONTEXT_FEATURES):
        stats = preprocessor["context_stats"][feature_name]
        values = context[:, feature_index]
        missing = context_missing[:, feature_index]
        values = np.where(missing, float(stats["median"]), values)
        if bool(stats.get("standardize", False)):
            values = (
                values - float(stats["mean"])
            ) / max(float(stats["std"]), 1e-8)
        context[:, feature_index] = values

    context = np.concatenate(
        [context, context_missing.astype(np.float32)],
        axis=1,
    )
    if not np.isfinite(token).all() or not np.isfinite(context).all():
        raise ValueError("Non-finite values remain after preprocessing.")
    return token.astype(np.float32), context.astype(np.float32)

__all__ = [
    "MODULES", "TOKEN_FEATURES", "CONTEXT_FEATURES", "HORIZONS",
    "MODULE_RANGES", "PatientVisit", "PatientHistory", "normalize_visits",
    "validate_patient_context", "build_patient_tensor_package",
    "transform_fold_features",
]
