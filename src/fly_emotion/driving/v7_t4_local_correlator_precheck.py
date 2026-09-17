"""Local-edge stop/go precheck for frozen normalized T4 correlators."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_conductance import _collect_source_normalization
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t4_normalized_correlator import (
    CONFIG as CORRELATOR_CONFIG,
)
from fly_emotion.driving.v7_t4_normalized_correlator import (
    IMPLEMENTATION as CORRELATOR_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4_normalized_correlator import (
    NormalizedCorrelatorT4Probe,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    CONFIG as LOCAL_EDGE_CONFIG,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    IMPLEMENTATION as LOCAL_EDGE_IMPLEMENTATION,
)
from fly_emotion.driving.v7_t4t5_local_edge_precheck import (
    evaluate_v7_t4t5_local_edge_precheck,
)

CONFIG = Path("configs/driving-v7-t4-local-correlator-precheck.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_local_correlator_precheck.py")


def evaluate_v7_t4_local_correlator_precheck(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    correlator_config = yaml.safe_load((root / CORRELATOR_CONFIG).read_text())
    conductance_path = Path(correlator_config["conductance_config"])
    conductance = yaml.safe_load((root / conductance_path).read_text())
    source_path = Path(config["source_audit_evidence"])
    source_audit = json.loads((root / source_path).read_text())
    normalization = _collect_source_normalization(
        root,
        retinal_backend=correlator_config["retinal_backend"],
        retinal_geometry=conductance["primary_retinal_geometry"],
        config=conductance,
    )
    candidates = {}
    for candidate in config["frozen_candidates"]:
        probe = NormalizedCorrelatorT4Probe(
            root,
            normalization,
            float(candidate["gain"]),
            str(candidate["mode"]),
            source_audit,
            lag_substeps=int(candidate["lag_substeps"]),
        )
        report = evaluate_v7_t4t5_local_edge_precheck(root, probe_override=probe)
        t4_scores = {
            name: score
            for name, score in report["population_scores"].items()
            if name.startswith("T4")
        }
        key = (
            f"{candidate['mode']}:gain={float(candidate['gain']):g}:"
            f"lag={int(candidate['lag_substeps'])}"
        )
        candidates[key] = {
            **candidate,
            "T4_direction_pass_count": sum(
                score["direction"]["passed"] for score in t4_scores.values()
            ),
            "T4_polarity_pass_count": sum(
                score["polarity"]["passed"] for score in t4_scores.values()
            ),
            "T4_population_scores": t4_scores,
        }
    minimum = int(config["precheck_gate"]["minimum_T4_direction_populations_to_expand"])
    advancing = [
        name for name, result in candidates.items() if result["T4_direction_pass_count"] >= minimum
    ]
    return {
        "protocol": {
            "name": config["name"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(LOCAL_EDGE_CONFIG): _sha256(root / LOCAL_EDGE_CONFIG),
                str(LOCAL_EDGE_IMPLEMENTATION): _sha256(root / LOCAL_EDGE_IMPLEMENTATION),
                str(CORRELATOR_CONFIG): _sha256(root / CORRELATOR_CONFIG),
                str(CORRELATOR_IMPLEMENTATION): _sha256(root / CORRELATOR_IMPLEMENTATION),
                str(conductance_path): _sha256(root / conductance_path),
                str(source_path): _sha256(root / source_path),
            },
            "candidate_count": len(candidates),
            "condition_id": "LPLC-T01",
            "parameter_fit": False,
            "target_activity_injection": False,
            "calibration_evaluated": False,
            "runtime_modified": False,
        },
        "candidates": candidates,
        "advancing_candidates": advancing,
        "local_correlator_gate_passed": bool(advancing),
        "expand_to_three_conditions": bool(advancing),
        "advance_to_calibration": False,
        "advance_to_runtime_integration": False,
        "boundary": config["boundary"],
    }
