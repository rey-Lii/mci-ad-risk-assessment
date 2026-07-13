import numpy as np
from ra_fmlr.evaluation import hazards_to_cumulative_risks, validate_monotonic_risks


def test_monotonic_risk():
    risks = hazards_to_cumulative_risks(np.array([[0.1, 0.2, 0.3, 0.4]]))
    validate_monotonic_risks(risks)
    assert np.all(np.diff(risks, axis=1) >= 0)
