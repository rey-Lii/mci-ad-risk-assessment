import pytest

from ra_fmlr.inference import predict_frozen_hybrid


VALID_VISITS = [{"date": "2026-01-01", "MMSE": 25}]


@pytest.mark.parametrize(
    ("age", "sex_male", "education_years", "message"),
    [
        (39, 1, 12, "Age"),
        (70, 2, 12, "Sex"),
        (70, 1, 31, "Education"),
    ],
)
def test_snapshot_route_rejects_invalid_demographics_before_loading_artifacts(
    age,
    sex_male,
    education_years,
    message,
):
    with pytest.raises(ValueError, match=message):
        predict_frozen_hybrid(
            age=age,
            sex_male=sex_male,
            education_years=education_years,
            visits=VALID_VISITS,
            ensemble=None,  # type: ignore[arg-type]
        )


def test_routed_inference_rejects_an_empty_visit_before_loading_artifacts():
    with pytest.raises(ValueError, match="at least one score"):
        predict_frozen_hybrid(
            age=70,
            sex_male=1,
            education_years=12,
            visits=[
                {"date": "2025-01-01", "MMSE": 27},
                {"date": "2026-01-01"},
            ],
            ensemble=None,  # type: ignore[arg-type]
        )
