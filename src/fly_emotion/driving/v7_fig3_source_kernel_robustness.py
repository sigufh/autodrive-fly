"""Robustness audit for empirical Fig. 3 source-type temporal kernels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-fig3-source-kernel-robustness.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_fig3_source_kernel_robustness.py"
)


def _unit(values: np.ndarray) -> np.ndarray:
    centered = values - values[0]
    return centered / max(float(np.linalg.norm(centered)), 1e-12)


def evaluate_v7_fig3_source_kernel_robustness(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source_protocol_path = Path(config["source_protocol"])
    source_protocol = yaml.safe_load(
        (root / source_protocol_path).read_text(encoding="utf-8")
    )
    source_evidence_path = Path(config["source_evidence"])
    source_evidence = json.loads(
        (root / source_evidence_path).read_text(encoding="utf-8")
    )
    workbook_path = root / source_protocol["source_data"]["path"]
    if _sha256(workbook_path) != source_evidence["protocol"]["source_file"][
        "sha256"
    ]:
        raise ValueError("Fig. 3 robustness audit source workbook is stale")
    baseline_start, baseline_stop = map(float, config["baseline_seconds"])
    gates = config["gates"]
    rng = np.random.default_rng(int(config["bootstrap"]["seed"]))
    conditions = {}
    for condition_name, condition in config["conditions"].items():
        frame = pd.read_excel(workbook_path, sheet_name=condition["sheet"])
        time = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        baseline_mask = (time >= baseline_start) & (time < baseline_stop)
        response_mask = (time >= float(condition["onset_seconds"])) & (
            time
            <= float(condition["onset_seconds"])
            + float(config["response_duration_seconds"])
        )
        results = {}
        for source_type in config["source_types"]:
            columns = [
                name for name in frame if str(name).startswith(f"{source_type}-")
            ]
            values = frame[columns].to_numpy(dtype=np.float64)
            baseline = np.mean(values[baseline_mask], axis=0)
            oriented = float(condition["sign"]) * (values[response_mask] - baseline)
            mean_kernel = _unit(np.mean(oriented, axis=1))
            median_kernel = _unit(np.median(oriented, axis=1))
            mean_median_correlation = float(np.corrcoef(mean_kernel, median_kernel)[0, 1])
            loo = []
            for cell_index in range(oriented.shape[1]):
                held_out = _unit(oriented[:, cell_index])
                training = _unit(
                    np.mean(np.delete(oriented, cell_index, axis=1), axis=1)
                )
                loo.append(float(np.corrcoef(held_out, training)[0, 1]))
            bootstrap = []
            for _ in range(int(config["bootstrap"]["replicates"])):
                first = np.mean(
                    oriented[
                        :, rng.choice(oriented.shape[1], oriented.shape[1], replace=True)
                    ],
                    axis=1,
                )
                second = np.mean(
                    oriented[
                        :, rng.choice(oriented.shape[1], oriented.shape[1], replace=True)
                    ],
                    axis=1,
                )
                bootstrap.append(float(np.corrcoef(_unit(first), _unit(second))[0, 1]))
            gate_values = {
                "mean_median_kernel_correlation": mean_median_correlation
                >= float(gates["minimum_mean_median_kernel_correlation"]),
                "median_leave_one_cell_out_correlation": float(np.median(loo))
                >= float(gates["minimum_median_leave_one_cell_out_correlation"]),
                "minimum_leave_one_cell_out_correlation": float(np.min(loo))
                >= float(gates["minimum_leave_one_cell_out_correlation"]),
                "bootstrap_split_correlation_p05": float(np.quantile(bootstrap, 0.05))
                >= float(gates["minimum_bootstrap_split_correlation_p05"]),
            }
            results[source_type] = {
                "cell_count": oriented.shape[1],
                "mean_median_kernel_correlation": mean_median_correlation,
                "median_leave_one_cell_out_correlation": float(np.median(loo)),
                "minimum_leave_one_cell_out_correlation": float(np.min(loo)),
                "negative_leave_one_cell_out_fraction": float(np.mean(np.asarray(loo) < 0.0)),
                "bootstrap_split_correlation_p05": float(np.quantile(bootstrap, 0.05)),
                "gates": gate_values,
                "passed": bool(all(gate_values.values())),
            }
        conditions[condition_name] = {
            "sources": results,
            "all_sources_passed": all(item["passed"] for item in results.values()),
        }
    passed = all(item["all_sources_passed"] for item in conditions.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(source_protocol_path): _sha256(root / source_protocol_path),
                str(source_evidence_path): _sha256(root / source_evidence_path),
                "pyproject.toml": _sha256(root / "pyproject.toml"),
            },
            "bootstrap_seed": int(config["bootstrap"]["seed"]),
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "conditions": conditions,
        "all_source_kernel_robustness_gates_passed": passed,
        "source_specific_kernel_candidate_authorized": False,
        "advance_to_functional_precheck": False,
        "stop_reason": "Fig3_source_kernels_not_robust_across_recorded_cells",
        "boundary": config["boundary"],
    }
