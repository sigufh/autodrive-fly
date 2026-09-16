from __future__ import annotations

import argparse
import json
from pathlib import Path

from .connectome.graph import build_canonical_graph, load_graph
from .connectome.overview import build_overview
from .connectome.pathways import build_pathways
from .data.audit import audit_annotations, audit_connections, audit_neurotransmitters, write_json
from .data.download import download_file
from .data.manifest import iter_files, load_manifest
from .driving.evaluate import (
    calibrate_stable_policy,
    evaluate_body_motor_interaction,
    evaluate_city_alpha,
    evaluate_constraints,
    evaluate_neural_decision_baseline,
    evaluate_neural_motor_adaptation,
    evaluate_neural_transfer,
    evaluate_panorama_release,
    evaluate_sensory_ablation,
    evaluate_sensory_gain_audit,
    train_neural_curriculum,
    train_sensory_pathway_curriculum,
    write_evaluation,
)
from .driving.v7 import (
    evaluate_v7_controlled_vision,
    evaluate_v7_optic_hex_axis_calibration,
    evaluate_v7_typed_visual_candidate,
    write_v7_manifest,
)
from .driving.v7_branched import evaluate_v7_branched_t4_candidate
from .driving.v7_closed_loop import (
    evaluate_v7_closed_loop_calibration,
    evaluate_v7_closed_loop_tuning,
)
from .driving.v7_closed_loop_controls import evaluate_v7_closed_loop_controls
from .driving.v7_closed_loop_multi import evaluate_v7_closed_loop_multi
from .driving.v7_conductance import evaluate_v7_published_conductance
from .driving.v7_coverage_response import evaluate_v7_coverage_response
from .driving.v7_danger_throttle import evaluate_v7_danger_throttle
from .driving.v7_disinhibition import evaluate_v7_conductance_order, evaluate_v7_disinhibition
from .driving.v7_ephys_audit import evaluate_v7_electrophysiology_audit
from .driving.v7_ephys_interface import evaluate_v7_ephys_interface
from .driving.v7_fc2_pfl_dna import evaluate_v7_fc2_pfl_dna
from .driving.v7_fig5_validation import evaluate_v7_fig5_validation
from .driving.v7_fit import fit_v7_t4_conductance
from .driving.v7_gain_audit import evaluate_v7_normalization_gain
from .driving.v7_geometry_sign import evaluate_v7_geometry_sign
from .driving.v7_goal_audit import evaluate_v7_goal_coverage
from .driving.v7_heading_ring import evaluate_v7_heading_ring
from .driving.v7_lamina_goal import evaluate_v7_lamina_goal
from .driving.v7_lamina_goal_symmetry import evaluate_v7_lamina_goal_symmetry
from .driving.v7_local_input_audit import (
    evaluate_v7_local_input_audit,
    evaluate_v7_receptor_mask_audit,
    evaluate_v7_t4_input_coverage,
)
from .driving.v7_looming_mechanism_audit import evaluate_v7_looming_mechanism_audit
from .driving.v7_lplc2_phenotype import evaluate_v7_lplc2_phenotype
from .driving.v7_mirror_audit import evaluate_v7_layerwise_mirror_audit
from .driving.v7_navigation_nested import evaluate_v7_navigation_nested
from .driving.v7_navigation_nested_eval import evaluate_v7_navigation_nested_candidate
from .driving.v7_nested_neural_screen import evaluate_v7_nested_neural_screen
from .driving.v7_neural_channel_controls import evaluate_v7_neural_channel_controls
from .driving.v7_neural_channels import (
    evaluate_v7_neural_channels_calibration,
    evaluate_v7_neural_channels_tuning,
)
from .driving.v7_neural_corridor import (
    evaluate_v7_local_column_corridor,
    evaluate_v7_neural_corridor,
)
from .driving.v7_neural_corridor_dagger import evaluate_v7_neural_corridor_dagger
from .driving.v7_neural_dynamics_local import evaluate_v7_neural_dynamics_local
from .driving.v7_neural_episode_controls import evaluate_v7_neural_episode_controls
from .driving.v7_neural_episode_cv import evaluate_v7_neural_episode_cv
from .driving.v7_neural_local_columns import evaluate_v7_neural_local_columns
from .driving.v7_neural_spectra import evaluate_v7_neural_spectra
from .driving.v7_neural_topology_controls import evaluate_v7_neural_topology_controls
from .driving.v7_perturbation import evaluate_v7_perturbation
from .driving.v7_phase_motion import evaluate_v7_phase_motion
from .driving.v7_pixel_sampling import evaluate_v7_pixel_sampling
from .driving.v7_r1r6_local import (
    evaluate_v7_r1r6_local_calibration,
    evaluate_v7_r1r6_local_tuning,
)
from .driving.v7_r1r6_local_controls import evaluate_v7_r1r6_local_controls
from .driving.v7_r1r6_multi import (
    evaluate_v7_r1r6_multi_calibration,
    evaluate_v7_r1r6_multi_tuning,
)
from .driving.v7_retina_audit import evaluate_v7_retina_column_audit
from .driving.v7_source_audit import evaluate_v7_t4_source_audit
from .driving.v7_spectral_controls import evaluate_v7_spectral_controls
from .driving.v7_stability import evaluate_v7_background_stability, evaluate_v7_feedback_cut
from .driving.v7_stage1_development import evaluate_v7_stage1_development
from .driving.v7_stage1_geometry_ab import evaluate_v7_stage1_geometry_ab
from .driving.v7_stage1_input_audit import evaluate_v7_stage1_input_audit
from .driving.v7_stage1_nested import evaluate_v7_stage1_nested
from .driving.v7_stage1_scoring import evaluate_v7_stage1_scoring
from .driving.v7_stage1_split import evaluate_v7_stage1_split
from .driving.v7_synchronous import evaluate_v7_synchronous_update
from .driving.v7_t4_source_resolved import evaluate_v7_t4_source_resolved
from .driving.v7_t5_conductance_audit import evaluate_v7_t5_conductance_audit
from .driving.v7_t5_data_audit import evaluate_v7_t5_data_audit
from .driving.v7_t5_label_audit import evaluate_v7_t5_label_audit
from .driving.v7_t5_phenotype import evaluate_v7_t5_phenotype
from .driving.v7_t5_spatial_order import evaluate_v7_t5_spatial_order
from .driving.v7_t5_supplement_audit import evaluate_v7_t5_supplement_audit
from .driving.v7_target_fit import evaluate_v7_target_fit_contract
from .driving.v7_temporal_audit import evaluate_v7_temporal_input_audit
from .driving.v7_timebase_audit import evaluate_v7_timebase_audit
from .driving.v7_visual_corridor_goal import evaluate_v7_visual_corridor_goal
from .driving.v7_visual_layer_locality import evaluate_v7_visual_layer_locality
from .driving.v7_visual_target_input_audit import evaluate_v7_visual_target_input_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autodrive-fly")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("configs/data-manifest.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset", action="append")
    download = subparsers.add_parser("download")
    download.add_argument("--dataset", action="append")
    download.add_argument("--include-optional", action="store_true")
    subparsers.add_parser("audit")
    subparsers.add_parser("build-graph")
    subparsers.add_parser("build-overview")
    subparsers.add_parser("build-pathways")
    evaluate_driving = subparsers.add_parser("evaluate-driving")
    evaluate_driving.add_argument("--train-episodes", type=int, default=48)
    evaluate_driving.add_argument("--evaluation-seeds", type=int, default=32)
    evaluate_driving.add_argument("--evaluation-start", type=int, default=400)
    calibrate = subparsers.add_parser("calibrate-policy")
    calibrate.add_argument("--episodes", type=int, default=48)
    calibrate.add_argument("--evaluation-seeds", type=int, default=32)
    calibrate.add_argument("--evaluation-start", type=int, default=400)
    constraints = subparsers.add_parser("evaluate-constraints")
    constraints.add_argument("--evaluation-seeds", type=int, default=32)
    constraints.add_argument("--evaluation-start", type=int, default=400)
    constraints.add_argument("--checkpoint", type=Path)
    constraints.add_argument("--output", type=Path)
    constraints.add_argument("--reference-report", type=Path)
    city = subparsers.add_parser("evaluate-city-alpha")
    city.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 7])
    neural = subparsers.add_parser("evaluate-neural-decision")
    neural.add_argument("--start", type=int, default=400)
    neural.add_argument("--count", type=int, default=8)
    train_neural = subparsers.add_parser("train-neural-v6")
    train_neural.add_argument("--train-episodes", type=int, default=24)
    train_neural.add_argument("--evaluation-start", type=int, default=600)
    train_neural.add_argument("--evaluation-seeds", type=int, default=8)
    train_neural.add_argument(
        "--stage",
        choices=["single", "triple", "nine"],
        default="single",
    )
    train_neural.add_argument("--resume", action="store_true")
    train_neural.add_argument(
        "--sensory-profile",
        choices=["front", "panorama", "panorama_flow", "panorama_flow_body"],
        default="front",
    )
    subparsers.add_parser("evaluate-neural-transfer")
    subparsers.add_parser("evaluate-neural-motor-adaptation")
    sensory = subparsers.add_parser("evaluate-sensory-ablation")
    sensory.add_argument("--train-episodes", type=int, default=4)
    sensory.add_argument("--evaluation-start", type=int, default=960)
    sensory.add_argument("--evaluation-seeds", type=int, default=4)
    gain_audit = subparsers.add_parser("evaluate-sensory-gains")
    gain_audit.add_argument("--seed", type=int, default=960)
    gain_audit.add_argument("--steps", type=int, default=120)
    body_interaction = subparsers.add_parser("evaluate-body-motor-interaction")
    body_interaction.add_argument("--start", type=int, default=960)
    body_interaction.add_argument("--count", type=int, default=4)
    body_interaction.add_argument("--body-gain", type=float, default=0.0003)
    sensory_train = subparsers.add_parser("train-sensory-pathway")
    sensory_train.add_argument("--episodes-per-stage", type=int, default=2)
    sensory_train.add_argument("--evaluation-start", type=int, default=980)
    sensory_train.add_argument("--evaluation-seeds", type=int, default=4)
    panorama_release = subparsers.add_parser("evaluate-panorama-release")
    panorama_release.add_argument("--start", type=int, default=1000)
    panorama_release.add_argument("--count", type=int, default=16)
    subparsers.add_parser("v7-init")
    subparsers.add_parser("v7-evaluate-vision")
    subparsers.add_parser("v7-evaluate-typed-vision")
    subparsers.add_parser("v7-evaluate-optic-axis")
    subparsers.add_parser("v7-audit-t4-sources")
    subparsers.add_parser("v7-evaluate-branched-t4")
    subparsers.add_parser("v7-evaluate-t4-conductance")
    subparsers.add_parser("v7-fit-t4-conductance")
    subparsers.add_parser("v7-audit-retina-columns")
    subparsers.add_parser("v7-audit-layer-mirror")
    subparsers.add_parser("v7-audit-temporal-input")
    subparsers.add_parser("v7-audit-local-input")
    subparsers.add_parser("v7-audit-receptor-mask")
    subparsers.add_parser("v7-audit-t4-input-coverage")
    subparsers.add_parser("v7-audit-coverage-response")
    subparsers.add_parser("v7-evaluate-mi9-sign")
    subparsers.add_parser("v7-evaluate-conductance-order")
    subparsers.add_parser("v7-audit-background-stability")
    subparsers.add_parser("v7-audit-feedback-cut")
    subparsers.add_parser("v7-evaluate-synchronous-update")
    subparsers.add_parser("v7-audit-normalization-gain")
    subparsers.add_parser("v7-audit-full-update-perturbation")
    subparsers.add_parser("v7-evaluate-phase-motion")
    subparsers.add_parser("v7-evaluate-geometry-sign")
    subparsers.add_parser("v7-evaluate-pixel-sampling")
    subparsers.add_parser("v7-audit-spectral-controls")
    subparsers.add_parser("v7-evaluate-neural-spectra")
    subparsers.add_parser("v7-audit-electrophysiology")
    subparsers.add_parser("v7-audit-timebase")
    subparsers.add_parser("v7-validate-published-fig5")
    subparsers.add_parser("v7-build-ephys-interface")
    subparsers.add_parser("v7-audit-t5-data")
    subparsers.add_parser("v7-audit-t5-conductance")
    subparsers.add_parser("v7-extract-t5-phenotype")
    subparsers.add_parser("v7-audit-goal-coverage")
    subparsers.add_parser("v7-audit-t5-labels")
    subparsers.add_parser("v7-freeze-stage1-split")
    subparsers.add_parser("v7-audit-t5-supplement")
    subparsers.add_parser("v7-freeze-stage1-scoring")
    subparsers.add_parser("v7-audit-stage1-input")
    subparsers.add_parser("v7-evaluate-stage1-development")
    subparsers.add_parser("v7-evaluate-stage1-geometry-ab")
    subparsers.add_parser("v7-audit-visual-target-inputs")
    subparsers.add_parser("v7-audit-looming-mechanisms")
    subparsers.add_parser("v7-freeze-stage1-nested")
    subparsers.add_parser("v7-freeze-target-fit")
    subparsers.add_parser("v7-evaluate-closed-loop-tuning")
    subparsers.add_parser("v7-evaluate-closed-loop-calibration")
    subparsers.add_parser("v7-evaluate-closed-loop-controls")
    subparsers.add_parser("v7-evaluate-closed-loop-multi")
    subparsers.add_parser("v7-evaluate-r1r6-multi-tuning")
    subparsers.add_parser("v7-evaluate-r1r6-multi-calibration")
    subparsers.add_parser("v7-evaluate-r1r6-local-tuning")
    subparsers.add_parser("v7-evaluate-r1r6-local-calibration")
    subparsers.add_parser("v7-evaluate-r1r6-local-controls")
    subparsers.add_parser("v7-evaluate-neural-channels-tuning")
    subparsers.add_parser("v7-evaluate-neural-channels-calibration")
    subparsers.add_parser("v7-evaluate-neural-channel-controls")
    subparsers.add_parser("v7-evaluate-neural-topology-controls")
    subparsers.add_parser("v7-evaluate-nested-neural-screen")
    subparsers.add_parser("v7-evaluate-t5-spatial-order")
    subparsers.add_parser("v7-evaluate-lplc2-phenotype")
    subparsers.add_parser("v7-evaluate-t4-source-resolved")
    subparsers.add_parser("v7-evaluate-heading-ring")
    subparsers.add_parser("v7-evaluate-neural-local-columns")
    subparsers.add_parser("v7-evaluate-neural-episode-cv")
    subparsers.add_parser("v7-evaluate-neural-episode-controls")
    subparsers.add_parser("v7-evaluate-fc2-pfl-dna")
    subparsers.add_parser("v7-freeze-navigation-nested")
    subparsers.add_parser("v7-evaluate-navigation-nested")
    subparsers.add_parser("v7-evaluate-danger-throttle")
    subparsers.add_parser("v7-evaluate-visual-corridor")
    subparsers.add_parser("v7-evaluate-neural-corridor")
    subparsers.add_parser("v7-evaluate-local-column-corridor")
    subparsers.add_parser("v7-evaluate-neural-dynamics-local")
    subparsers.add_parser("v7-evaluate-neural-corridor-dagger")
    subparsers.add_parser("v7-evaluate-visual-layer-locality")
    subparsers.add_parser("v7-evaluate-lamina-goal")
    subparsers.add_parser("v7-evaluate-lamina-goal-symmetry")
    train_neural.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "evaluate-driving":
        report = write_evaluation(
            root,
            root / "artifacts/driving-evaluation.json",
            train_episodes=args.train_episodes,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
        )
        print(
            f"frozen {report['frozen']['mean_distance']:.2f} m -> "
            f"learned {report['learned']['mean_distance']:.2f} m "
            f"(delta {report['delta_mean_distance']:+.2f} m)"
        )
        return
    if args.command == "calibrate-policy":
        report = calibrate_stable_policy(
            root,
            episodes=args.episodes,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
        )
        target = root / "artifacts/stable-policy-calibration.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-constraints":
        report = evaluate_constraints(
            root,
            evaluation_seeds=args.evaluation_seeds,
            evaluation_start=args.evaluation_start,
            checkpoint=root / args.checkpoint if args.checkpoint else None,
            reference_report=root / args.reference_report if args.reference_report else None,
        )
        target = args.output or root / "artifacts/behavior-constraint-ablation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        delta = report["paired"]["distance"]["delta_mean"]
        print(f"lane-constraint distance delta {delta:+.2f} m -> {target}")
        return
    if args.command == "evaluate-city-alpha":
        report = evaluate_city_alpha(root, seeds=tuple(args.seeds))
        target = root / "artifacts/city-alpha-evaluation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-decision":
        report = evaluate_neural_decision_baseline(root, start=args.start, count=args.count)
        target = root / "artifacts/neural-decision-current.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "train-neural-v6":
        report = train_neural_curriculum(
            root,
            train_episodes=args.train_episodes,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
            publish=args.publish,
            stage=args.stage,
            resume=args.resume,
            sensory_profile=args.sensory_profile,
        )
        profile_suffix = "" if args.sensory_profile == "front" else f"-{args.sensory_profile}"
        target = root / f"artifacts/neural-v6-{args.stage}{profile_suffix}-curriculum.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-transfer":
        report = evaluate_neural_transfer(root)
        target = root / "artifacts/neural-v6-transfer.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-neural-motor-adaptation":
        report = evaluate_neural_motor_adaptation(root)
        target = root / "artifacts/neural-v6-motor-adaptation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-sensory-ablation":
        report = evaluate_sensory_ablation(
            root,
            train_episodes=args.train_episodes,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
        )
        target = root / "artifacts/neural-v6-sensory-ablation.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-sensory-gains":
        report = evaluate_sensory_gain_audit(root, seed=args.seed, steps=args.steps)
        target = root / "artifacts/neural-v6-sensory-gain-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-body-motor-interaction":
        report = evaluate_body_motor_interaction(
            root, start=args.start, count=args.count, body_gain=args.body_gain
        )
        target = root / "artifacts/neural-v6-body-motor-interaction.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "train-sensory-pathway":
        report = train_sensory_pathway_curriculum(
            root,
            episodes_per_stage=args.episodes_per_stage,
            evaluation_start=args.evaluation_start,
            evaluation_seeds=args.evaluation_seeds,
        )
        target = root / "artifacts/neural-v6-sensory-pathway-curriculum.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "evaluate-panorama-release":
        report = evaluate_panorama_release(root, start=args.start, count=args.count)
        target = root / "artifacts/neural-v6-panorama-release.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-init":
        report = write_v7_manifest(root)
        target = root / "artifacts/v7-manifest.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-vision":
        report = evaluate_v7_controlled_vision(root)
        target = root / "artifacts/v7-controlled-vision.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-typed-vision":
        report = evaluate_v7_typed_visual_candidate(root)
        target = root / "artifacts/v7-typed-visual-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-optic-axis":
        report = evaluate_v7_optic_hex_axis_calibration(root)
        target = root / "artifacts/v7-optic-hex-axis-calibration.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-sources":
        report = evaluate_v7_t4_source_audit(root)
        target = root / "artifacts/v7-t4-source-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-branched-t4":
        report = evaluate_v7_branched_t4_candidate(root)
        target = root / "artifacts/v7-branched-t4-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-conductance":
        report = evaluate_v7_published_conductance(root)
        target = root / "artifacts/v7-t4-conductance-candidate.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-fit-t4-conductance":
        report = fit_v7_t4_conductance(root)
        target = root / "artifacts/v7-t4-conductance-fit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-retina-columns":
        report = evaluate_v7_retina_column_audit(root)
        target = root / "artifacts/v7-retina-column-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-layer-mirror":
        report = evaluate_v7_layerwise_mirror_audit(root)
        target = root / "artifacts/v7-layerwise-mirror-audit.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-temporal-input":
        report = evaluate_v7_temporal_input_audit(root)
        target = root / "artifacts/v7-temporal-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-local-input":
        report = evaluate_v7_local_input_audit(root)
        target = root / "artifacts/v7-local-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-receptor-mask":
        report = evaluate_v7_receptor_mask_audit(root)
        target = root / "artifacts/v7-receptor-mask-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t4-input-coverage":
        report = evaluate_v7_t4_input_coverage(root)
        target = root / "artifacts/v7-t4-input-coverage.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-coverage-response":
        report = evaluate_v7_coverage_response(root)
        target = root / "artifacts/v7-t4-coverage-response.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-mi9-sign":
        report = evaluate_v7_disinhibition(root)
        target = root / "artifacts/v7-mi9-sign-comparison.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-conductance-order":
        report = evaluate_v7_conductance_order(root)
        target = root / "artifacts/v7-conductance-order-comparison.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-background-stability":
        report = evaluate_v7_background_stability(root)
        target = root / "artifacts/v7-background-stability.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-feedback-cut":
        report = evaluate_v7_feedback_cut(root)
        target = root / "artifacts/v7-feedback-cut.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-synchronous-update":
        report = evaluate_v7_synchronous_update(root)
        target = root / "artifacts/v7-synchronous-update.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-normalization-gain":
        report = evaluate_v7_normalization_gain(root)
        target = root / "artifacts/v7-normalization-gain.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-full-update-perturbation":
        report = evaluate_v7_perturbation(root)
        target = root / "artifacts/v7-full-update-perturbation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-data":
        report = evaluate_v7_t5_data_audit(root)
        target = root / "artifacts/v7-t5-data-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-conductance":
        report = evaluate_v7_t5_conductance_audit(root)
        target = root / "artifacts/v7-t5-conductance-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-extract-t5-phenotype":
        report = evaluate_v7_t5_phenotype(root)
        target = root / "artifacts/v7-t5-phenotype.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-goal-coverage":
        report = evaluate_v7_goal_coverage(root)
        target = root / "artifacts/v7-goal-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-labels":
        report = evaluate_v7_t5_label_audit(root)
        target = root / "artifacts/v7-t5-label-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-split":
        report = evaluate_v7_stage1_split(root)
        target = root / "artifacts/v7-stage1-split.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-t5-supplement":
        report = evaluate_v7_t5_supplement_audit(root)
        target = root / "artifacts/v7-t5-supplement-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-scoring":
        report = evaluate_v7_stage1_scoring(root)
        target = root / "artifacts/v7-stage1-scoring.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-stage1-input":
        report = evaluate_v7_stage1_input_audit(root)
        target = root / "artifacts/v7-stage1-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-stage1-development":
        report = evaluate_v7_stage1_development(root)
        target = root / "artifacts/v7-stage1-development.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-stage1-geometry-ab":
        report = evaluate_v7_stage1_geometry_ab(root)
        target = root / "artifacts/v7-stage1-geometry-ab.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-visual-target-inputs":
        report = evaluate_v7_visual_target_input_audit(root)
        target = root / "artifacts/v7-visual-target-input-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-looming-mechanisms":
        report = evaluate_v7_looming_mechanism_audit(root)
        target = root / "artifacts/v7-looming-mechanism-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-stage1-nested":
        report = evaluate_v7_stage1_nested(root)
        target = root / "artifacts/v7-stage1-nested.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-target-fit":
        report = evaluate_v7_target_fit_contract(root)
        target = root / "artifacts/v7-target-fit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-tuning":
        report = evaluate_v7_closed_loop_tuning(root)
        target = root / "artifacts/v7-closed-loop-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-calibration":
        report = evaluate_v7_closed_loop_calibration(root)
        target = root / "artifacts/v7-closed-loop-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-controls":
        report = evaluate_v7_closed_loop_controls(root)
        target = root / "artifacts/v7-closed-loop-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-closed-loop-multi":
        report = evaluate_v7_closed_loop_multi(root)
        target = root / "artifacts/v7-closed-loop-multi.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-multi-tuning":
        report = evaluate_v7_r1r6_multi_tuning(root)
        target = root / "artifacts/v7-r1r6-multi-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-multi-calibration":
        report = evaluate_v7_r1r6_multi_calibration(root)
        target = root / "artifacts/v7-r1r6-multi-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-tuning":
        report = evaluate_v7_r1r6_local_tuning(root)
        target = root / "artifacts/v7-r1r6-local-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-calibration":
        report = evaluate_v7_r1r6_local_calibration(root)
        target = root / "artifacts/v7-r1r6-local-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-r1r6-local-controls":
        report = evaluate_v7_r1r6_local_controls(root)
        target = root / "artifacts/v7-r1r6-local-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channels-tuning":
        report = evaluate_v7_neural_channels_tuning(root)
        target = root / "artifacts/v7-neural-channels-tuning.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channels-calibration":
        report = evaluate_v7_neural_channels_calibration(root)
        target = root / "artifacts/v7-neural-channels-calibration.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-channel-controls":
        report = evaluate_v7_neural_channel_controls(root)
        target = root / "artifacts/v7-neural-channel-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-topology-controls":
        report = evaluate_v7_neural_topology_controls(root)
        target = root / "artifacts/v7-neural-topology-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-nested-neural-screen":
        report = evaluate_v7_nested_neural_screen(root)
        target = root / "artifacts/v7-nested-neural-screen.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t5-spatial-order":
        report = evaluate_v7_t5_spatial_order(root)
        target = root / "artifacts/v7-t5-spatial-order.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lplc2-phenotype":
        report = evaluate_v7_lplc2_phenotype(root)
        target = root / "artifacts/v7-lplc2-phenotype.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-t4-source-resolved":
        report = evaluate_v7_t4_source_resolved(root)
        target = root / "artifacts/v7-t4-source-resolved.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-heading-ring":
        report = evaluate_v7_heading_ring(root)
        target = root / "artifacts/v7-heading-ring.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-local-columns":
        report = evaluate_v7_neural_local_columns(root)
        target = root / "artifacts/v7-neural-local-columns.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-episode-cv":
        report = evaluate_v7_neural_episode_cv(root)
        target = root / "artifacts/v7-neural-episode-cv.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-episode-controls":
        report = evaluate_v7_neural_episode_controls(root)
        target = root / "artifacts/v7-neural-episode-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-fc2-pfl-dna":
        report = evaluate_v7_fc2_pfl_dna(root)
        target = root / "artifacts/v7-fc2-pfl-dna.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-freeze-navigation-nested":
        report = evaluate_v7_navigation_nested(root)
        target = root / "artifacts/v7-navigation-nested.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-navigation-nested":
        report = evaluate_v7_navigation_nested_candidate(root)
        target = root / "artifacts/v7-navigation-nested-eval.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-danger-throttle":
        report = evaluate_v7_danger_throttle(root)
        target = root / "artifacts/v7-danger-throttle.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-visual-corridor":
        report = evaluate_v7_visual_corridor_goal(root)
        target = root / "artifacts/v7-visual-corridor-goal.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-corridor":
        report = evaluate_v7_neural_corridor(root)
        target = root / "artifacts/v7-neural-corridor.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-local-column-corridor":
        report = evaluate_v7_local_column_corridor(root)
        target = root / "artifacts/v7-local-column-corridor.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-dynamics-local":
        report = evaluate_v7_neural_dynamics_local(root)
        target = root / "artifacts/v7-neural-dynamics-local.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-corridor-dagger":
        report = evaluate_v7_neural_corridor_dagger(root)
        target = root / "artifacts/v7-neural-corridor-dagger.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-visual-layer-locality":
        report = evaluate_v7_visual_layer_locality(root)
        target = root / "artifacts/v7-visual-layer-locality.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lamina-goal":
        report = evaluate_v7_lamina_goal(root)
        target = root / "artifacts/v7-lamina-goal.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-lamina-goal-symmetry":
        report = evaluate_v7_lamina_goal_symmetry(root)
        target = root / "artifacts/v7-lamina-goal-symmetry.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-build-ephys-interface":
        report = evaluate_v7_ephys_interface(root)
        target = root / "artifacts/v7-ephys-interface.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-validate-published-fig5":
        report = evaluate_v7_fig5_validation(root)
        target = root / "artifacts/v7-fig5-validation.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-timebase":
        report = evaluate_v7_timebase_audit(root)
        target = root / "artifacts/v7-timebase-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-electrophysiology":
        report = evaluate_v7_electrophysiology_audit(root)
        target = root / "artifacts/v7-electrophysiology-audit.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-neural-spectra":
        report = evaluate_v7_neural_spectra(root)
        target = root / "artifacts/v7-neural-spectra.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-audit-spectral-controls":
        report = evaluate_v7_spectral_controls(root)
        target = root / "artifacts/v7-spectral-controls.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-pixel-sampling":
        report = evaluate_v7_pixel_sampling(root)
        target = root / "artifacts/v7-pixel-sampling.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-geometry-sign":
        report = evaluate_v7_geometry_sign(root)
        target = root / "artifacts/v7-geometry-sign.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    if args.command == "v7-evaluate-phase-motion":
        report = evaluate_v7_phase_motion(root)
        target = root / "artifacts/v7-phase-motion.json"
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(target)
        return
    manifest = load_manifest(root / args.manifest)
    if args.command in {"verify", "download"}:
        selected = set(args.dataset) if args.dataset else None
        failures: list[str] = []
        for spec in iter_files(manifest, selected):
            if spec.optional and not getattr(args, "include_optional", False):
                continue
            if args.command == "download":
                print(f"downloading {spec.dataset}/{spec.name} -> {spec.path}")
                download_file(spec, root)
            errors = spec.verify(root)
            print(f"{'OK' if not errors else 'FAIL'} {spec.dataset}/{spec.name}")
            failures.extend(errors)
        if failures:
            raise SystemExit("\n".join(failures))
        return

    annotations = root / "data/raw/malecns-v1.0/body-annotations.feather"
    if args.command == "build-graph":
        graph = build_canonical_graph(
            annotations,
            root / "data/raw/malecns-v1.0/connectome-weights.feather",
            root / "data/processed/malecns-v1.0",
        )
        print(f"built {graph.node_count} nodes and {graph.edge_count} edges")
        return
    if args.command == "build-overview":
        overview = build_overview(annotations, root / "data/processed/malecns-v1.0/overview.json")
        print(f"built overview for {overview['positioned_nodes']} positioned neurons")
        return
    if args.command == "build-pathways":
        graph = load_graph(root / "data/processed/malecns-v1.0", normalized=False)
        pathways = build_pathways(
            graph, annotations, root / "data/processed/malecns-v1.0/pathways.json"
        )
        print(
            f"built {len(pathways['bundled_paths'])} bundles and "
            f"{len(pathways['strong_paths'])} strong paths from "
            f"{pathways['all_edges_accounted']} edges"
        )
        return
    transmitters = root / "data/raw/malecns-v1.0/body-neurotransmitters.feather"
    annotation_report, canonical = audit_annotations(annotations)
    report = {
        "malecns": {
            "annotations": annotation_report,
            "neurotransmitters": audit_neurotransmitters(transmitters, canonical),
            "connections": audit_connections(
                root / "data/raw/malecns-v1.0/connectome-weights.feather", canonical
            ),
        }
    }
    expected = manifest["datasets"]["malecns"]
    observed = report["malecns"]
    checks = {
        "expected_canonical_nodes": observed["annotations"]["canonical_nodes"],
        "expected_source_connection_rows": observed["connections"]["source_rows"],
        "expected_canonical_edges": observed["connections"]["canonical_edges"],
        "expected_canonical_synapse_weight_sum": observed["connections"][
            "canonical_synapse_weight_sum"
        ],
    }
    mismatches = {
        key: (expected[key], value) for key, value in checks.items() if expected[key] != value
    }
    if mismatches:
        raise ValueError(f"MaleCNS manifest count mismatch: {mismatches}")
    target = root / "artifacts/data-audit.json"
    write_json(report, target)
    print(target)


if __name__ == "__main__":
    main()
