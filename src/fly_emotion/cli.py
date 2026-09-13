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
    evaluate_city_alpha,
    evaluate_constraints,
    evaluate_neural_decision_baseline,
    write_evaluation,
)


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
        target = root / "artifacts/neural-decision-baseline.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
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
