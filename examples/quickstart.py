"""Synthetic examples: input validation, tensors, and routing only."""

from ra_fmlr import PatientHistory, PatientVisit, select_route


def describe(patient: PatientHistory) -> None:
    package = patient.build_tensor_package()
    print("route:", select_route(patient.history_depth))
    print("history depth:", patient.history_depth)
    print("token shape:", package["token_features_raw"].shape)
    print("module availability:", package["module_available"].tolist())
    print()


single_visit = PatientHistory(
    age=73,
    sex_male=0,
    education_years=14,
    visits=(PatientVisit(date="2026-01-01", MMSE=25, CDGLOBAL=0.5, CDRSB=2, FAQTOTAL=4),),
)

longitudinal = PatientHistory(
    age=73,
    sex_male=0,
    education_years=14,
    visits=(
        PatientVisit(date="2025-01-01", MMSE=27, CDGLOBAL=0.5, CDRSB=1, FAQTOTAL=2),
        PatientVisit(date="2026-01-01", MMSE=25, CDGLOBAL=0.5, CDRSB=2, FAQTOTAL=4),
    ),
)


if __name__ == "__main__":
    describe(single_visit)
    describe(longitudinal)
    print("No hard-coded risks are returned. Private frozen artifacts are required for prediction.")
