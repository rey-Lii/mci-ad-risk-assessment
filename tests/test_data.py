from datetime import date, datetime

import numpy as np
import pytest

from ra_fmlr import PatientHistory, PatientVisit
from ra_fmlr.data import build_patient_tensor_package, normalize_visits


def test_tensor_contract():
    patient = PatientHistory(
        age=70,
        sex_male=1,
        education_years=12,
        visits=(
            PatientVisit(date="2025-01-01", MMSE=27, CDRSB=1),
            PatientVisit(date="2026-01-01", MMSE=25, CDRSB=2),
        ),
    )
    package = patient.build_tensor_package()
    assert package["token_features_raw"].shape == (1, 5, 16, 8)
    assert patient.history_depth == 2


def test_visits_are_sorted_chronologically():
    visits = normalize_visits(
        [
            {"date": "2026-01-01", "MMSE": 25},
            {"date": "2025-01-01", "MMSE": 27},
        ]
    )
    assert [visit["date"].year for visit in visits] == [2025, 2026]


def test_mixed_date_types_on_same_day_are_rejected():
    patient = PatientHistory(
        age=70,
        sex_male=1,
        education_years=12,
        visits=(
            PatientVisit(date=date(2026, 1, 1), MMSE=27),
            PatientVisit(date=datetime(2026, 1, 1), MMSE=25),
        ),
    )
    with pytest.raises(ValueError, match="different date"):
        _ = patient.history_depth


def test_each_visit_requires_at_least_one_score():
    with pytest.raises(ValueError, match="at least one score"):
        normalize_visits(
            [
                {"date": "2025-01-01", "MMSE": 27},
                {"date": "2026-01-01"},
            ]
        )


@pytest.mark.parametrize(
    ("age", "sex_male", "education_years", "message"),
    [
        (39, 1, 12, "Age"),
        (111, 1, 12, "Age"),
        (70, 1.5, 12, "Sex"),
        (70, 1, -1, "Education"),
        (70, 1, 31, "Education"),
    ],
)
def test_invalid_demographics_are_rejected(
    age,
    sex_male,
    education_years,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_patient_tensor_package(
            age=age,
            sex_male=sex_male,
            education_years=education_years,
            visits=[{"date": "2026-01-01", "MMSE": 25}],
        )


def test_delta_slope_gap_and_recency_are_constructed_correctly():
    package = build_patient_tensor_package(
        age=70,
        sex_male=1,
        education_years=12,
        visits=[
            {"date": "2025-01-01", "MMSE": 28},
            {"date": "2026-01-01", "MMSE": 26},
        ],
    )

    mmse = package["token_features_raw"][0, 1]
    expected_slope = -2.0 / 365.0 * 365.25

    assert mmse[0, 0] == pytest.approx(28.0)
    assert mmse[1, 0] == pytest.approx(26.0)
    assert mmse[1, 1] == pytest.approx(-2.0)
    assert mmse[1, 2] == pytest.approx(expected_slope)
    assert mmse[1, 4] == pytest.approx(365.0 / 365.25)
    assert mmse[0, 5] == pytest.approx(365.0 / 365.25)
    assert mmse[1, 5] == pytest.approx(0.0)
    assert np.array_equal(
        package["observation_mask"][0, 1, :2],
        np.array([1, 1], dtype=np.uint8),
    )
