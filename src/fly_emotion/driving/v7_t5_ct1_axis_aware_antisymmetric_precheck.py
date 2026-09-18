"""Corrected antisymmetric T5 CT1 cross-fit axis precheck."""

from pathlib import Path

from fly_emotion.driving.v7_t5_ct1_axis_aware_precheck import _evaluate

CONFIG = Path(
    "configs/driving-v7-t5-ct1-axis-aware-antisymmetric-precheck.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_ct1_axis_aware_antisymmetric_precheck.py"
)


def evaluate_v7_t5_ct1_axis_aware_antisymmetric_precheck(root: Path) -> dict:
    return _evaluate(root, CONFIG, IMPLEMENTATION)
