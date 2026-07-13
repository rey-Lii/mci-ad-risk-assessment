from ra_fmlr import PatientHistory, PatientVisit


def test_tensor_contract():
    patient = PatientHistory(
        age=70, sex_male=1, education_years=12,
        visits=(
            PatientVisit(date="2025-01-01", MMSE=27, CDRSB=1),
            PatientVisit(date="2026-01-01", MMSE=25, CDRSB=2),
        ),
    )
    package = patient.build_tensor_package()
    assert package["token_features_raw"].shape == (1, 5, 16, 8)
    assert patient.history_depth == 2
