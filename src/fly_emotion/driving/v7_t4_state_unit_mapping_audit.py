"""Audit the published Fig. 3 voltage normalization as a v7 state mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-state-unit-mapping-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_state_unit_mapping_audit.py")


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _ordinals(ids: list[str], source: str) -> list[int]:
    prefix = f"{source}-"
    if any(not value.startswith(prefix) for value in ids):
        raise ValueError(f"unexpected individual ID for {source}")
    return [int(value[len(prefix) :]) - 1 for value in ids]


def evaluate_v7_t4_state_unit_mapping_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if not all(config["held_out_protocol"].values()):
        raise ValueError("T4 state-unit held-out protocol must remain strict")
    retrieval_path = Path(config["retrieval_evidence"])
    split_path = Path(config["individual_split_evidence"])
    contract_path = Path(config["v7_contract"])
    runtime_path = Path(config["v7_implementation"])
    engine_path = Path(config["runtime_engine"])
    retrieval = json.loads((root / retrieval_path).read_text(encoding="utf-8"))
    split = json.loads((root / split_path).read_text(encoding="utf-8"))
    notebook_path = root / config["notebook"]["path"]
    if (
        notebook_path.stat().st_size != int(config["notebook"]["bytes"])
        or _sha256(notebook_path) != config["notebook"]["sha256"]
    ):
        raise ValueError("T4 Fig. 3 notebook changed")
    notebook_source = _notebook_source(notebook_path)
    author_snippets = (
        "def normalize(a):",
        "anorm=a-np.min(a)",
        "anorm=anorm/np.max(anorm)",
        "Mi1_ = np.nanmean(Mi1a, axis=1)",
        "Mi4_ = np.nanmean(Mi4a, axis=1)",
    )
    if not all(snippet in notebook_source for snippet in author_snippets):
        raise ValueError("T4 Fig. 3 normalization semantics changed")
    runtime_source = (root / runtime_path).read_text(encoding="utf-8")
    runtime_snippets = (
        "np.tanh(1.8 * recurrent + drive)",
        "state = np.zeros(self.graph.node_count, dtype=np.float32)",
    )
    if not all(snippet in runtime_source for snippet in runtime_snippets):
        raise ValueError("v7 runtime state semantics changed")
    if '"simulated_activation_not_millivolts"' not in (root / engine_path).read_text(
        encoding="utf-8"
    ):
        raise ValueError("runtime state unit disclosure changed")
    if not retrieval["gates"]["all_four_local_payloads_hash_and_structure_verified"]:
        raise ValueError("verified 1-kHz T4 source payloads are required")
    if not split["frozen_split_contract"]["every_individual_appears_in_validation_at_least_once"]:
        raise ValueError("complete held-out individual coverage is required")

    source_results = {}
    for source in config["source_order"]:
        filename = config["source_files"][source]
        expected = retrieval["frozen_file_manifest"][filename]
        path = root / config["source_directory"] / filename
        if path.stat().st_size != int(expected["bytes"]) or _sha256(path) != expected["sha256"]:
            raise ValueError(f"T4 source payload changed: {filename}")
        values = np.load(path, allow_pickle=False)
        if list(values.shape) != expected["shape"] or str(values.dtype) != expected["dtype"]:
            raise ValueError(f"T4 source payload structure changed: {filename}")
        population_mean = np.mean(values, axis=1)
        full_minimum = float(np.min(population_mean))
        full_maximum = float(np.max(population_mean))
        full_normalized = (population_mean - full_minimum) / (full_maximum - full_minimum)
        minimum_index = np.unravel_index(np.argmin(population_mean), population_mean.shape)
        maximum_index = np.unravel_index(np.argmax(population_mean), population_mean.shape)

        fold_results = []
        folds = split["conditions"]["on"]["sources"][source]["folds"]
        off_folds = split["conditions"]["off"]["sources"][source]["folds"]
        if [
            (item["training_individual_ids"], item["validation_individual_ids"]) for item in folds
        ] != [
            (item["training_individual_ids"], item["validation_individual_ids"])
            for item in off_folds
        ]:
            raise ValueError("T4 ON/OFF individual split folds differ")
        for fold in folds:
            training = _ordinals(fold["training_individual_ids"], source)
            validation = _ordinals(fold["validation_individual_ids"], source)
            if set(training) & set(validation):
                raise ValueError("T4 unit mapping fold is not individual-disjoint")
            training_mean = np.mean(values[:, training, :], axis=1)
            validation_mean = np.mean(values[:, validation, :], axis=1)
            training_minimum = float(np.min(training_mean))
            training_maximum = float(np.max(training_mean))
            mapped = (validation_mean - training_minimum) / (training_maximum - training_minimum)
            below = int(np.count_nonzero(mapped < 0.0))
            above = int(np.count_nonzero(mapped > 1.0))
            fold_results.append(
                {
                    "fold_index": int(fold["fold_index"]),
                    "training_individual_count": len(training),
                    "validation_individual_count": len(validation),
                    "training_validation_disjoint": True,
                    "training_minimum_millivolts": training_minimum,
                    "training_maximum_millivolts": training_maximum,
                    "validation_mapped_minimum": float(np.min(mapped)),
                    "validation_mapped_maximum": float(np.max(mapped)),
                    "validation_samples_below_zero": below,
                    "validation_samples_above_one": above,
                    "validation_sample_count": int(mapped.size),
                    "all_validation_samples_in_author_interval": below + above == 0,
                }
            )
        outside = sum(
            item["validation_samples_below_zero"] + item["validation_samples_above_one"]
            for item in fold_results
        )
        validation_count = sum(item["validation_sample_count"] for item in fold_results)
        training_constant_pairs = {
            (
                item["training_minimum_millivolts"],
                item["training_maximum_millivolts"],
            )
            for item in fold_results
        }
        source_results[source] = {
            "cell_count": int(values.shape[1]),
            "author_full_cohort_constants": {
                "minimum_millivolts": full_minimum,
                "maximum_millivolts": full_maximum,
                "minimum_condition": config["stimulus_order"][int(minimum_index[0])],
                "minimum_sample": int(minimum_index[1]),
                "maximum_condition": config["stimulus_order"][int(maximum_index[0])],
                "maximum_sample": int(maximum_index[1]),
                "normalized_minimum": float(np.min(full_normalized)),
                "normalized_maximum": float(np.max(full_normalized)),
            },
            "held_out_folds": fold_results,
            "held_out_fold_count": len(fold_results),
            "distinct_training_constant_pair_count": len(training_constant_pairs),
            "training_constants_identical_across_folds": (len(training_constant_pairs) == 1),
            "held_out_folds_with_below_zero_samples": sum(
                item["validation_samples_below_zero"] > 0 for item in fold_results
            ),
            "held_out_folds_with_above_one_samples": sum(
                item["validation_samples_above_one"] > 0 for item in fold_results
            ),
            "held_out_samples_outside_author_interval": outside,
            "held_out_sample_count": validation_count,
            "held_out_outside_interval_fraction": outside / validation_count,
            "held_out_mapped_minimum": min(
                item["validation_mapped_minimum"] for item in fold_results
            ),
            "held_out_mapped_maximum": max(
                item["validation_mapped_maximum"] for item in fold_results
            ),
            "all_held_out_samples_in_author_interval": outside == 0,
        }

    author_formula_reproduced = all(
        item["author_full_cohort_constants"]["normalized_minimum"] == 0.0
        and item["author_full_cohort_constants"]["normalized_maximum"] == 1.0
        for item in source_results.values()
    )
    all_held_out_in_interval = all(
        item["all_held_out_samples_in_author_interval"] for item in source_results.values()
    )
    one_training_mapping_per_fold = all(
        all(
            item["training_maximum_millivolts"] > item["training_minimum_millivolts"]
            for item in source["held_out_folds"]
        )
        for source in source_results.values()
    )
    deployment_training_cohort_declared = False
    gate_values = {
        "author_formula_reproduced": author_formula_reproduced,
        "every_held_out_sample_in_author_interval": all_held_out_in_interval,
        "single_training_fitted_mapping_per_fold": one_training_mapping_per_fold,
        "deployment_training_cohort_declared": deployment_training_cohort_declared,
        "author_mapping_semantics_match_v7_state": False,
        "held_out_clipping_rule_available": False,
    }
    transfer_authorized = all(gate_values.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(retrieval_path): _sha256(root / retrieval_path),
                str(split_path): _sha256(root / split_path),
                str(contract_path): _sha256(root / contract_path),
                str(runtime_path): _sha256(root / runtime_path),
                str(engine_path): _sha256(root / engine_path),
                str(config["notebook"]["path"]): _sha256(notebook_path),
                **{
                    str(Path(config["source_directory"]) / filename): _sha256(
                        root / config["source_directory"] / filename
                    )
                    for filename in config["source_files"].values()
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "author_mapping": {
            "formula": config["author_formula"]["expression"],
            "scope": config["author_formula"]["scope"],
            "output_interval": config["author_formula"]["output_interval"],
            "formula_reproduced_on_full_cohort": author_formula_reproduced,
            "uses_both_ON_and_OFF_conditions_to_define_constants": True,
            "held_out_clipping_rule_declared": False,
        },
        "v7_state_semantics": {
            "activation_function": "signed_tanh",
            "initial_state": 0.0,
            "reported_unit": "simulated_activation_not_millivolts",
            "author_zero_to_one_voltage_semantics_match_v7_state": False,
        },
        "source_results": source_results,
        "gates": gate_values,
        "millivolts_to_v7_normalized_state_mapping_available": transfer_authorized,
        "authorize_source_dynamics_transfer": False,
        "authorize_runtime_integration": False,
        "stop_reason": "author_full_cohort_minmax_not_a_deployable_v7_state_mapping",
        "boundary": config["boundary"],
    }
