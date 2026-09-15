from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7 import V7_CONFIG, V7_IMPLEMENTATION, V7Contract, _sha256

CONFIG = Path("configs/driving-v7-goal-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_goal_audit.py")


def _load_evidence(root: Path, paths: dict[str, str]) -> tuple[dict[str, dict], dict[str, str]]:
    reports = {}
    hashes = {}
    for name, relative in paths.items():
        path = root / relative
        if not path.exists():
            raise ValueError(f"required goal-audit evidence is missing: {relative}")
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        hashes[relative] = _sha256(path)
    return reports, hashes


def evaluate_v7_goal_coverage(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract = V7Contract.load(root)
    reports, evidence_hashes = _load_evidence(root, config["evidence"])
    if contract.payload["stage_order"] != config["required_stage_order"]:
        raise ValueError("v7 stage order differs from objective audit")
    manifest = reports["manifest"]
    controlled = reports["controlled_vision"]
    t5 = reports["t5_phenotype"]
    t5_labels = reports["t5_label_audit"]
    interface = reports["ephys_interface"]
    stage1_split = reports["stage1_split"]
    stage1_scoring = reports["stage1_scoring"]
    stage1_input = reports["stage1_input"]
    t5_supplement = reports["t5_supplement"]
    baseline_hashes_valid = all(
        _sha256(root / item["path"]) == item["sha256"]
        for item in contract.payload["baseline_contracts"].values()
    )
    visual_inputs_only_at_receptors = (
        controlled["protocol"]["direct_input_type"] == "R1-R6"
        and controlled["protocol"]["target_direct_input_overlap"] == 0
    )
    visual_response_pass = bool(
        reports["typed_vision"]["controlled_response_gates_pass"]
        and reports["branched_t4"]["controlled_response_gates_pass"]
        and reports["conductance_t4"]["controlled_response_gates_pass"]
        and reports["fitted_t4"]["validation_passed"]
    )
    topology_complete = bool(
        controlled["stage1_topology_controls_complete"]
        and controlled["strict_degree_preserving_control_complete"]
        and controlled["real_topology_advantage"]
    )
    stage1_pass = visual_response_pass and topology_complete
    if manifest["advance_to_central_complex"] != stage1_pass:
        raise ValueError("manifest and independent coverage audit disagree on stage 1")
    downstream_authorized = bool(stage1_pass)
    checks = [
        {
            "item": 0,
            "deliverable": "independent v5/v6/v7 versions, checkpoints and claims",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
            "observations": {
                "baseline_hashes_valid": baseline_hashes_valid,
                "v7_deployment_enabled": contract.payload["deployment_enabled"],
                "v7_city_expansion_enabled": contract.payload["city_expansion_enabled"],
                "default_runtime_changed": manifest["default_runtime_changed"],
            },
        },
        {
            "item": 1,
            "deliverable": "R1-R6 causal controlled vision through T4/T5 and LPLC/LC",
            "status": "failed",
            "evidence": [
                config["evidence"]["controlled_vision"],
                config["evidence"]["typed_vision"],
                config["evidence"]["branched_t4"],
                config["evidence"]["conductance_t4"],
                config["evidence"]["fitted_t4"],
                config["evidence"]["t5_phenotype"],
            ],
            "observations": {
                "stimulus_count": controlled["protocol"]["stimulus_count"],
                "stimulus_families": sorted({item["family"] for item in controlled["stimuli"]}),
                "visual_inputs_only_at_receptors": visual_inputs_only_at_receptors,
                "controlled_response_gates_pass": visual_response_pass,
                "T5_measured_pair_count": t5["summary"]["all_pairs"]["pair_count"],
                "T5_biological_PD_code_assigned": t5["label_boundary"][
                    "biological_PD_code_assigned"
                ],
                "T5_external_direction_label_map_verified": t5_labels["label_status"][
                    "direction_code_to_PD_ND_mapping_verified"
                ],
                "T5_model_scoring_allowed": t5["label_boundary"]["model_scoring_allowed"],
                "T5_published_fitted_parameter_vectors_available": t5_supplement["replay_status"][
                    "published_17_cell_parameter_values_available"
                ],
                "physical_timebase_identified": reports["timebase"]["identifiability"][
                    "physical_timebase_identified"
                ],
                "strict_scoring_contract_frozen": all(
                    stage1_scoring["synthetic_controls"].values()
                ),
                "development_R1_R6_input_gates_pass": stage1_input["input_gates_pass"],
                "development_neural_evaluation_allowed": stage1_input[
                    "development_neural_evaluation_allowed"
                ],
                "existing_artifacts_rescored_under_new_contract": stage1_scoring["protocol"][
                    "existing_artifacts_rescored"
                ],
            },
            "missing": [
                "complete T4/T5 direction and ON/OFF response gates",
                "LPLC1/LPLC2/LC4 looming and collision response gates",
                "an independent T5 label map suitable for model scoring",
                "a calibrated physical visual/neural timebase",
            ],
        },
        {
            "item": 2,
            "deliverable": "EPG/PEN/PEG heading, occlusion memory and FC2/PFL comparison",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
            "observations": {"stage1_pass": stage1_pass},
        },
        {
            "item": 3,
            "deliverable": "DNp20, DNa01/DNa02 and PFL3-to-DNa readout comparison",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
            "observations": {"stage1_pass": stage1_pass},
        },
        {
            "item": 4,
            "deliverable": (
                "separate self-motion, looming, near-collision, heading and target signals"
            ),
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
            "observations": {"stage1_pass": stage1_pass},
        },
        {
            "item": 5,
            "deliverable": "KC-MBON value learning gated by PAM/PPL1",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
            "observations": {"causal_visual_navigation_pass": False},
        },
        {
            "item": 6,
            "deliverable": (
                "real topology, strict graph shuffles, parametric baselines and ablations"
            ),
            "status": "incomplete",
            "evidence": [config["evidence"]["controlled_vision"]],
            "observations": {
                "preliminary_stage1_controls_complete": controlled[
                    "stage1_topology_controls_complete"
                ],
                "strict_degree_preserving_control_complete": controlled[
                    "strict_degree_preserving_control_complete"
                ],
                "real_topology_advantage": controlled["real_topology_advantage"],
                "parameter_matched_linear_mlp_gru_complete": False,
                "downstream_ablations_authorized": downstream_authorized,
            },
        },
        {
            "item": 7,
            "deliverable": "independent train/validation/final and dynamic OOD release gates",
            "status": "not_authorized",
            "evidence": [
                config["evidence"]["fitted_t4"],
                config["evidence"]["ephys_interface"],
            ],
            "observations": {
                "fitted_T4_final_test_evaluated": reports["fitted_t4"]["test"]["evaluated"],
                "ephys_final_test_available": interface["available_final_test"],
                "task_and_OOD_release_evaluated": False,
                "stage1_stimulus_splits_disjoint": all(
                    value["identity_overlap"] == value["frame_hash_overlap"] == 0
                    for value in stage1_split["cross_split_overlap"].values()
                ),
                "reserved_final_evaluated": stage1_split["split_manifests"]["final"]["evaluated"],
                "blinded_one_time_final_available": False,
            },
        },
        {
            "item": 8,
            "deliverable": "pause city/LLM expansion and separate planner, fly core and executor",
            "status": "passed_boundary_only",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
            "observations": {
                "city_expansion_enabled": contract.payload["city_expansion_enabled"],
                "v7_deployment_enabled": contract.payload["deployment_enabled"],
                "three_way_contribution_report_available": False,
            },
        },
    ]
    requirement_checklist = [
        {
            "requirement": "0.version_config_evidence_claim_boundaries",
            "status": "passed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "0.preserve_assisted_v5_checkpoint",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [contract.payload["baseline_contracts"]["assisted_v5"]["path"]],
        },
        {
            "requirement": "0.preserve_neural_v6_front_checkpoint",
            "status": "passed" if baseline_hashes_valid else "failed",
            "evidence": [contract.payload["baseline_contracts"]["neural_v6_front"]["path"]],
        },
        {
            "requirement": "0.keep_default_service_unchanged",
            "status": "passed" if not manifest["default_runtime_changed"] else "failed",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "1.controlled_brightness_on_off_motion_looming_flow_mirror_stimuli",
            "status": "passed",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "1.external_visual_input_only_via_R1_R6",
            "status": "passed" if visual_inputs_only_at_receptors else "failed",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "1.T4_T5_direction_and_ON_OFF_validation",
            "status": "failed",
            "evidence": [config["evidence"]["typed_vision"], config["evidence"]["t5_phenotype"]],
        },
        {
            "requirement": "1.LPLC1_LPLC2_LC4_approach_collision_validation",
            "status": "failed",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "2.EPG_PEN_PEG_heading_and_occlusion",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "2.FC2_PFL3_PFL2_heading_goal_and_turn_termination",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "3.DNp20_DNa01_DNa02_PFL3_DNa_readout_comparison",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "3.transparent_environment_blind_vehicle_mapping",
            "status": "present_in_v5_v6_not_yet_validated_for_v7",
            "evidence": ["src/fly_emotion/driving/engine.py"],
        },
        {
            "requirement": "4.separate_self_motion_looming_slowing_heading_target_channels",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "5.KC_MBON_PAM_PPL1_value_learning_after_navigation_gate",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "6.real_MaleCNS_and_strict_randomized_graph_controls",
            "status": "incomplete",
            "evidence": [config["evidence"]["controlled_vision"]],
        },
        {
            "requirement": "6.parameter_matched_linear_MLP_GRU_baselines",
            "status": "not_started",
            "evidence": [],
        },
        {
            "requirement": "6.T4_T5_LPLC_EPG_PFL_DNa_MBON_DAN_ablations",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": "7.independent_train_validation_one_time_final_split",
            "status": "protocol_frozen_not_evaluated",
            "evidence": [
                config["evidence"]["fitted_t4"],
                config["evidence"]["ephys_interface"],
                config["evidence"]["stage1_split"],
                config["evidence"]["stage1_scoring"],
            ],
        },
        {
            "requirement": "7.dynamic_obstacle_density_curvature_speed_noise_OOD",
            "status": "not_authorized",
            "evidence": [config["evidence"]["manifest"]],
        },
        {
            "requirement": (
                "7.task_collision_exit_window_early_failure_steering_drift_"
                "mirror_zero_action_topology_gates"
            ),
            "status": "not_authorized",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "8.pause_city_and_large_model_expansion",
            "status": "passed",
            "evidence": [str(V7_CONFIG), config["evidence"]["manifest"]],
        },
        {
            "requirement": "8.separate_planner_fly_core_executor_contributions",
            "status": "missing",
            "evidence": [],
        },
        {
            "requirement": "8.continuous_tests_evidence_docs_frontend_and_remote_updates",
            "status": "in_progress",
            "evidence": [
                "tests",
                "docs/driving-v7.zh-CN.md",
                "apps/web/src/components/DrivingPanel.tsx",
            ],
        },
        {
            "requirement": "8.preserve_negative_results_and_stop_conditions",
            "status": "passed",
            "evidence": [config["evidence"]["manifest"], config["evidence"]["t5_phenotype"]],
        },
    ]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(V7_CONFIG): contract.sha256,
                str(V7_IMPLEMENTATION): _sha256(root / V7_IMPLEMENTATION),
                **evidence_hashes,
            },
        },
        "objective_items": config["objective_items"],
        "checks": checks,
        "requirement_checklist": requirement_checklist,
        "summary": {
            "complete_items": [item["item"] for item in checks if item["status"] == "passed"],
            "boundary_only_items": [
                item["item"] for item in checks if item["status"] == "passed_boundary_only"
            ],
            "failed_items": [item["item"] for item in checks if item["status"] == "failed"],
            "incomplete_items": [item["item"] for item in checks if item["status"] == "incomplete"],
            "not_authorized_items": [
                item["item"] for item in checks if item["status"] == "not_authorized"
            ],
            "objective_complete": False,
            "current_stage": "controlled_vision",
            "next_allowed_work": [
                "verify an external direction-code map before using T5 traces to score a model",
                "repair T4/T5/LPLC/LC visual dynamics without target-state injection",
                "evaluate only development under the frozen strict scoring contract",
                "obtain externally custodied independent-cell and one-time final manifests",
            ],
        },
    }
