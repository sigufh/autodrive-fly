"""Audit author baseline and aggregation semantics for measured T5 kernels."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_kohn_portes_t5_source_kernel_audit import (
    _author_rescaled_temporal,
)

CONFIG = Path(
    "configs/driving-v7-t5-population-kernel-aggregation-semantics-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_t5_population_kernel_aggregation_semantics_audit.py"
)


def _code_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _return_mean_temporal_calls(path: Path) -> list[dict]:
    calls = []
    for line_number, line in enumerate(_code_text(path).splitlines(), start=1):
        if "return_mean_temporal(" not in line:
            continue
        try:
            tree = ast.parse(line.strip())
        except SyntaxError as error:
            raise ValueError(
                f"unparseable return_mean_temporal call at {path}:{line_number}"
            ) from error
        node = next(
            (
                item
                for item in ast.walk(tree)
                if isinstance(item, ast.Call)
                and (
                    (isinstance(item.func, ast.Name) and item.func.id == "return_mean_temporal")
                    or (
                        isinstance(item.func, ast.Attribute)
                        and item.func.attr == "return_mean_temporal"
                    )
                )
            ),
            None,
        )
        if node is None:
            continue
        positional_baseline = ast.unparse(node.args[1]) if len(node.args) > 1 else None
        keyword = next((item for item in node.keywords if item.arg == "baseline"), None)
        calls.append(
            {
                "line": line_number,
                "source": ast.unparse(node),
                "baseline_argument": (
                    ast.unparse(keyword.value) if keyword is not None else positional_baseline
                ),
            }
        )
    return sorted(calls, key=lambda item: item["line"])


def _default_baseline_semantics(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "return_mean_temporal"
    )
    arguments = [argument.arg for argument in function.args.args]
    defaults = [None] * (len(arguments) - len(function.args.defaults)) + list(
        function.args.defaults
    )
    baseline_default = defaults[arguments.index("baseline")]
    default_is_none = isinstance(baseline_default, ast.Constant) and baseline_default.value is None

    def baseline_branch(value: str) -> ast.If:
        return next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and ast.unparse(node.test) == f"baseline == {value!r}"
        )

    def assigns(body: list[ast.stmt], expression: str) -> bool:
        return any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "b" for target in node.targets)
            and ast.unparse(node.value) == expression
            for node in body
        )

    start_branch = baseline_branch("start")
    end_branch = baseline_branch("end")
    population_mean_is_row_weighted = any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "np.mean"
        and len(node.args) == 2
        and ast.unparse(node.args[0]) == "av"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 0
        for node in ast.walk(function)
    )
    return {
        "baseline_default_is_none": default_is_none,
        "start_baseline_uses_first_sample": assigns(start_branch.body, "trace[0]"),
        "end_baseline_uses_last_100_sample_mean": assigns(
            end_branch.body, "np.mean(trace[-100:])"
        ),
        "default_branch_uses_zero_offset": assigns(end_branch.orelse, "0"),
        "population_mean_is_row_weighted": population_mean_is_row_weighted,
    }


def evaluate_v7_t5_population_kernel_aggregation_semantics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_path = Path(config["source_kernel_evidence"])
    source_config_path = Path(config["source_config"])
    measured_config_path = Path(config["measured_kernel_config"])
    measured_implementation_path = Path(config["measured_kernel_implementation"])
    evidence = json.loads((root / evidence_path).read_text(encoding="utf-8"))
    source_config = yaml.safe_load((root / source_config_path).read_text(encoding="utf-8"))
    measured_config = yaml.safe_load(
        (root / measured_config_path).read_text(encoding="utf-8")
    )
    if list(evidence["source_results"]) != config["source_order"]:
        raise ValueError("source-kernel evidence order changed")
    if evidence["aggregate"]["kernel_lengths"] != [499, 500]:
        raise ValueError("source-kernel length inventory changed")
    returnmeans_path = _verify_file(root, config["author_returnmeans"])
    semantics = _default_baseline_semantics(returnmeans_path)
    if not all(semantics.values()):
        raise ValueError("author return_mean_temporal semantics changed")
    call_sites = {}
    total_calls = 0
    explicit_baseline_calls = 0
    call_paths = {}
    for name, spec in config["author_call_sites"].items():
        path = root / spec["path"]
        if not path.is_file():
            raise ValueError(f"missing bounded author call-site file: {path}")
        calls = _return_mean_temporal_calls(path)
        if len(calls) != int(spec["expected_call_count"]):
            raise ValueError(f"return_mean_temporal call inventory changed: {name}")
        explicit = sum(call["baseline_argument"] is not None for call in calls)
        call_sites[name] = {
            "path": spec["path"],
            "call_count": len(calls),
            "calls_with_explicit_baseline": explicit,
            "calls": calls,
        }
        call_paths[name] = path
        total_calls += len(calls)
        explicit_baseline_calls += explicit

    source_results = {}
    source_paths = {}
    for source in config["source_order"]:
        spec = source_config["white_noise_files"][source]
        path = _verify_file(root, spec)
        source_paths[source] = path
        records = _load_restricted(path)
        row_kernels = np.stack(
            [
                _author_rescaled_temporal(record, int(config["kernel_length"]))
                for record in records
            ]
        )
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for record, kernel in zip(records, row_kernels, strict=True):
            grouped[str(record["recording_id"])].append(kernel)
        author_row_mean = np.mean(row_kernels, axis=0)
        equal_id_mean = np.mean(
            [np.mean(kernels, axis=0) for kernels in grouped.values()], axis=0
        )
        author_row_mean /= np.sum(np.abs(author_row_mean))
        equal_id_mean /= np.sum(np.abs(equal_id_mean))
        source_results[source] = {
            "payload_row_count": len(records),
            "unique_recording_id_count": len(grouped),
            "duplicate_recording_id_row_count": len(records) - len(grouped),
            "author_row_weighted_equals_v7_equal_recording_id_weighted": bool(
                np.array_equal(author_row_mean, equal_id_mean)
            ),
            "row_weighted_vs_equal_recording_id_correlation": float(
                np.corrcoef(author_row_mean, equal_id_mean)[0, 1]
            ),
            "row_weighted_vs_equal_recording_id_maximum_absolute_difference": float(
                np.max(np.abs(author_row_mean - equal_id_mean))
            ),
        }
    author_default_used = bool(total_calls > 0 and explicit_baseline_calls == 0)
    current_no_baseline = measured_config["kernel"].get("baseline_subtraction") is None
    exact_aggregation = all(
        item["author_row_weighted_equals_v7_equal_recording_id_weighted"]
        for item in source_results.values()
    )
    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(evidence_path): _sha256(root / evidence_path),
        str(source_config_path): _sha256(root / source_config_path),
        str(measured_config_path): _sha256(root / measured_config_path),
        str(measured_implementation_path): _sha256(root / measured_implementation_path),
        config["author_returnmeans"]["path"]: _sha256(returnmeans_path),
        **{
            spec["path"]: _sha256(call_paths[name])
            for name, spec in config["author_call_sites"].items()
        },
        **{
            source_config["white_noise_files"][source]["path"]: _sha256(path)
            for source, path in source_paths.items()
        },
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "author_function_semantics": semantics,
        "bounded_call_sites": call_sites,
        "call_summary": {
            "call_count": total_calls,
            "calls_with_explicit_baseline": explicit_baseline_calls,
            "all_observed_calls_use_default_baseline_none": author_default_used,
        },
        "current_v7_semantics": {
            "baseline_subtraction": None,
            "matches_author_default_baseline_semantics": bool(
                author_default_used and current_no_baseline
            ),
            "aggregation": measured_config["kernel"]["source_summary"],
            "matches_author_row_weighted_population_mean_exactly": exact_aggregation,
            "deliberate_recording_id_balancing": True,
        },
        "source_results": source_results,
        "author_default_baseline_semantics_verified": author_default_used,
        "current_no_baseline_matches_author_default": bool(
            author_default_used and current_no_baseline
        ),
        "author_row_weighted_aggregation_exactly_reproduced": exact_aggregation,
        "alternative_tail_baseline_variant_authorized": False,
        "population_kernel_transfer_authorized": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "author_default_baseline_reproduced_but_recording_balancing_differs_"
            "and_transfer_gates_remain_closed"
        ),
        "boundary": config["boundary"],
    }
