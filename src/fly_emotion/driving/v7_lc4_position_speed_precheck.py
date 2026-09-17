"""Position-covered LC4 angular-speed stop/go precheck."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import VisualStimulus
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_lplc2_position_coverage import _filled_disc
from fly_emotion.driving.v7_lplc2_radial_opponency import _layer_traces
from fly_emotion.driving.v7_nested_neural_screen import MassBalancedVisualProbe

CONFIG = Path("configs/driving-v7-lc4-position-speed-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_lc4_position_speed_precheck.py")


def evaluate_v7_lc4_position_speed_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    position_path = Path(config["position_protocol"])
    position = yaml.safe_load((root / position_path).read_text())
    typed_path = Path(config["typed_stimulus_protocol"])
    typed = yaml.safe_load((root / typed_path).read_text())
    condition = next(
        row for row in typed["conditions"] if row["condition_id"] == config["condition_id"]
    )
    if condition["role"] != "tuning":
        raise ValueError("LC4 speed precheck may consume tuning only")
    probe = MassBalancedVisualProbe(root, config)
    populations = {side: probe.populations[f"LC4_{side}"] for side in ("L", "R")}
    centers = [
        (float(x), float(y)) for y in position["centers"]["y"] for x in position["centers"]["x"]
    ]
    baseline = np.full((int(config["common_background_frames"]), 24, 48), 0.92, dtype=np.float32)
    base_duration = int(condition["duration_frames"])
    responses = {side: [] for side in populations}
    durations = []
    for divisor in config["duration_divisors"]:
        duration = max(4, base_duration // int(divisor))
        durations.append(duration)
        by_position = {side: [] for side in populations}
        for center_x, center_y in centers:
            frames = _filled_disc(
                duration,
                float(condition["start_radius_pixels"]),
                float(condition["terminal_radius_pixels"]),
                center_x,
                center_y,
                True,
            )
            stimulus = VisualStimulus(
                f"LC4:x={center_x:g}:y={center_y:g}:duration={duration}",
                "filled_looming",
                "off",
                "expansion",
                np.concatenate((baseline, frames)),
            )
            traces = _layer_traces(probe, stimulus, populations)
            for side in populations:
                by_position[side].append(np.max(traces[side], axis=0))
        for side in populations:
            responses[side].append(np.max(np.stack(by_position[side]), axis=0))

    velocities = np.asarray(config["relative_angular_velocities"], dtype=np.float64)
    centered_velocity = velocities - velocities.mean()
    results = {}
    for side, per_speed in responses.items():
        matrix = np.stack(per_speed)
        centered = matrix - matrix.mean(axis=0)
        slopes = centered_velocity @ centered / float(centered_velocity @ centered_velocity)
        fitted = matrix.mean(axis=0) + centered_velocity[:, None] * slopes[None, :]
        residual = np.sum((matrix - fitted) ** 2, axis=0)
        total = np.sum(centered**2, axis=0)
        valid = np.isfinite(slopes) & (
            total >= float(config["thresholds"]["minimum_valid_denominator"])
        )
        r_squared = np.full(len(slopes), np.nan, dtype=np.float64)
        r_squared[valid] = 1.0 - residual[valid] / total[valid]
        positive_fraction = float(np.mean(slopes[valid] > 0.0)) if np.any(valid) else 0.0
        monotonic = (matrix[2] > matrix[1]) & (matrix[1] > matrix[0])
        median_r_squared = float(np.median(r_squared[valid])) if np.any(valid) else None
        gates = {
            "valid_cell_fraction": float(np.mean(valid))
            >= float(config["thresholds"]["minimum_valid_cell_fraction"]),
            "positive_slope_fraction": positive_fraction
            >= float(config["thresholds"]["minimum_positive_slope_fraction"]),
            "median_r_squared": median_r_squared is not None
            and median_r_squared >= float(config["thresholds"]["minimum_median_r_squared"]),
            "monotonic_cell_fraction": float(np.mean(monotonic))
            >= float(config["thresholds"]["minimum_monotonic_cell_fraction"]),
        }
        results[side] = {
            "cell_count": int(len(slopes)),
            "valid_cell_count": int(np.count_nonzero(valid)),
            "valid_cell_fraction": float(np.mean(valid)),
            "positive_slope_fraction": positive_fraction,
            "median_r_squared": median_r_squared,
            "monotonic_cell_count": int(np.count_nonzero(monotonic)),
            "monotonic_cell_fraction": float(np.mean(monotonic)),
            "median_response_by_speed": np.median(matrix, axis=1).tolist(),
            "gates": gates,
            "passed": all(gates.values()),
        }
    passed = all(result["passed"] for result in results.values())
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(position_path): _sha256(root / position_path),
                str(typed_path): _sha256(root / typed_path),
            },
            "condition_id": config["condition_id"],
            "position_count": len(centers),
            "duration_frames": durations,
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "per_side": results,
        "LC4_position_speed_precheck_passed": bool(passed),
        "expand_to_three_conditions": bool(passed),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
