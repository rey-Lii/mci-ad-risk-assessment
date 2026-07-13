"""Wrappers for the final frozen training stages.

Full retraining requires the complete archived pipeline and authorized data.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

FINAL_STAGES = {
    "snapshot": "18n_train_history1_resource_aware_logistic_cdrsb_qc.py",
    "transformer": "18o_train_v61_gap_controlled_latest_anchored_cdrsb_qc.py",
    "assemble": "18p_build_final_history_adaptive_hybrid_oof_cdrsb_qc.py",
    "evaluate": "18q_final_hybrid_ipcw_calibration_bootstrap.py",
}


def run_stage(stage: str, full_pipeline_root: Path) -> None:
    if stage not in FINAL_STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    root = full_pipeline_root.expanduser().resolve()
    script = root / "scripts" / FINAL_STAGES[stage]
    if not script.is_file():
        raise FileNotFoundError(script)
    subprocess.run(
        [sys.executable, str(script), "--project-root", str(root)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(FINAL_STAGES))
    parser.add_argument("--full-pipeline-root", type=Path, required=True)
    args = parser.parse_args()
    run_stage(args.stage, args.full_pipeline_root)


if __name__ == "__main__":
    main()
