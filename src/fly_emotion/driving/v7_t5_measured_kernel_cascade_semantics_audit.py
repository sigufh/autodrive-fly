"""Audit the discrete source-state-to-measured-kernel cascade semantics."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_lamina_split import LaminaSplitProbe

CONFIG = Path("configs/driving-v7-t5-measured-kernel-cascade-semantics-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_measured_kernel_cascade_semantics_audit.py"
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(function: ast.FunctionDef) -> list[str]:
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def evaluate_v7_t5_measured_kernel_cascade_semantics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paths = {
        name: Path(config[name])
        for name in (
            "measured_kernel_evidence",
            "measured_kernel_config",
            "measured_kernel_implementation",
            "lamina_split_config",
            "lamina_split_implementation",
            "nested_probe_implementation",
            "v7_contract",
            "v7_implementation",
            "state_mapping_evidence",
        )
    }
    measured_evidence = json.loads(
        (root / paths["measured_kernel_evidence"]).read_text(encoding="utf-8")
    )
    measured_config = yaml.safe_load(
        (root / paths["measured_kernel_config"]).read_text(encoding="utf-8")
    )
    lamina = yaml.safe_load(
        (root / paths["lamina_split_config"]).read_text(encoding="utf-8")
    )
    v7 = yaml.safe_load((root / paths["v7_contract"]).read_text(encoding="utf-8"))
    state_mapping = json.loads(
        (root / paths["state_mapping_evidence"]).read_text(encoding="utf-8")
    )
    expected = config["expected"]
    actual = {
        "retinal_backend": lamina["retinal_backend"],
        "dynamics_backend": lamina["dynamics_backend"],
        "baseline_frames": int(lamina["baseline_frames"]),
        "brain_updates_per_frame": int(lamina["brain_substeps_per_frame"]),
        "source_types": measured_config["source_types"],
        "source_leaks": {
            source: float(v7["controlled_vision"]["typed_visual_leak_v1"][source])
            for source in measured_config["source_types"]
        },
    }
    if actual != expected:
        raise ValueError("measured-kernel cascade configuration changed")
    if measured_evidence["kernel_summary"]["source_order"] != expected[
        "source_types"
    ]:
        raise ValueError("measured-kernel evidence source order changed")
    if (
        measured_evidence["boundary"][
            "source_activity_driven_only_from_R1_R6_propagation"
        ]
        is not True
    ):
        raise ValueError("measured-kernel evidence input boundary changed")

    measured_path = root / paths["measured_kernel_implementation"]
    v7_path = root / paths["v7_implementation"]
    source_sequence = _function(measured_path, "_source_sequences")
    candidate_traces = _function(measured_path, "_candidate_traces")
    advance = _function(v7_path, "_advance")
    retinal_code = _function(v7_path, "_retinal_code")
    sequence_source = ast.unparse(source_sequence)
    advance_source = ast.unparse(advance)
    retinal_source = ast.unparse(retinal_code)
    source_calls = _call_names(source_sequence)
    candidate_calls = _call_names(candidate_traces)
    gates = {
        "retinal_drive_uses_previous_frame_difference": (
            "values - previous" in retinal_source and "np.clip" in retinal_source
        ),
        "source_sequence_calls_probe_advance": "_advance" in source_calls,
        "probe_advance_contains_recurrent_signed_adjacency": (
            "self.adjacency @ (transmitted * self.source_sign)" in advance_source
        ),
        "probe_advance_contains_tanh_and_leak": (
            "np.tanh" in advance_source and "self.leak" in advance_source
        ),
        "source_sequence_subtracts_recurrent_baseline_state": (
            "state.astype(np.float64) - baseline_state" in sequence_source
        ),
        "source_sequence_half_wave_rectifies_before_moments": (
            "np.maximum(state.astype(np.float64) - baseline_state, 0.0)"
            in sequence_source
        ),
        "measured_kernel_convolution_occurs_after_source_sequence": (
            "_convolve" in candidate_calls
        ),
        "discrete_delay_backend_inactive": lamina["dynamics_backend"]
        != "columnar_delay_v1",
        "target_correlator_backend_inactive": lamina["dynamics_backend"]
        != "columnar_correlator_v1",
        "source_state_mapping_available": state_mapping[
            "millivolts_or_filter_output_to_v7_state_mapping_available"
        ],
    }
    probe = LaminaSplitProbe(root, lamina)
    source_nodes = {
        source: np.flatnonzero(probe.node_types == source)
        for source in measured_config["source_types"]
    }
    runtime_leaks = {
        source: sorted(np.unique(probe.leak[nodes]).astype(float).tolist())
        for source, nodes in source_nodes.items()
    }
    leak_matches = all(
        len(runtime_leaks[source]) == 1
        and np.isclose(runtime_leaks[source][0], expected["source_leaks"][source])
        for source in expected["source_types"]
    )
    if not leak_matches:
        raise ValueError("runtime source leak values changed")
    cascade_present = all(
        gates[name]
        for name in (
            "retinal_drive_uses_previous_frame_difference",
            "source_sequence_calls_probe_advance",
            "probe_advance_contains_recurrent_signed_adjacency",
            "probe_advance_contains_tanh_and_leak",
            "source_sequence_subtracts_recurrent_baseline_state",
            "source_sequence_half_wave_rectifies_before_moments",
            "measured_kernel_convolution_occurs_after_source_sequence",
        )
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in paths.values()},
            },
            "parameter_fit": False,
            "functional_stimulus_evaluated": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "configured_path": actual,
        "runtime_source_node_counts": {
            source: len(nodes) for source, nodes in source_nodes.items()
        },
        "runtime_source_leaks": runtime_leaks,
        "runtime_source_leaks_match_config": leak_matches,
        "semantic_gates": gates,
        "typed_recurrent_source_state_then_measured_kernel_cascade_verified": (
            cascade_present
        ),
        "measured_kernel_replaces_existing_source_dynamics": False,
        "single_stage_biological_source_model_interpretation_authorized": False,
        "external_recording_to_v7_source_state_mapping_available": gates[
            "source_state_mapping_available"
        ],
        "authorize_physical_source_dynamics_transfer": False,
        "authorize_T5_functional_precheck": False,
        "direction_scoring_authorized": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "measured_kernel_is_cascaded_after_uncalibrated_typed_recurrent_source_state"
        ),
        "boundary": config["boundary"],
    }
