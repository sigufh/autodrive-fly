"""Audit the per-source scope of T5 source-to-MaleCNS mappings."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t5-source-mapping-scope-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_source_mapping_scope_audit.py")


def evaluate_v7_t5_source_mapping_scope_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {
        name: Path(config[name])
        for name in (
            "source_kernel_evidence",
            "type_average_mapping_evidence",
            "malecns_mapping_evidence",
            "ct1_columnar_evidence",
        )
    }
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    kernels = evidence["source_kernel_evidence"]
    type_average = evidence["type_average_mapping_evidence"]
    malecns = evidence["malecns_mapping_evidence"]
    ct1_columnar = evidence["ct1_columnar_evidence"]
    required = config["required_sources"]
    expected = config["expected"]
    kernel_sources = list(kernels["source_results"])
    if kernel_sources != expected["voltage_derived_kernel_sources"]:
        raise ValueError("T5 voltage-derived source-kernel coverage changed")
    mapped = [
        source
        for source in required
        if type_average["source_mappings"][source]["mapping_contract_complete"]
    ]
    if mapped != expected["exact_type_average_sources"]:
        raise ValueError("T5 exact-type-average mapping scope changed")
    if ct1_columnar["CT1_complete_official_LO_column_coverage"]:
        raise ValueError("CT1 column coverage unexpectedly became complete")
    ct1_body_ids = sorted(int(value) for value in malecns["CT1"]["body_ids"])
    if ct1_body_ids != expected["CT1_body_ids"]:
        raise ValueError("CT1 body IDs changed")
    ct1_counts = {
        body_id: int(item["Lo1_column_union"]["column_count"])
        for body_id, item in ct1_columnar["CT1_bodies"].items()
    }
    if ct1_counts != expected["CT1_Lo1_column_counts"]:
        raise ValueError("CT1 Lo1 column counts changed")

    rows = {}
    for source in required:
        mapping = type_average["source_mappings"][source]
        anatomy = malecns["source_mapping"][source]
        gates = {
            "voltage_derived_source_kernel_available": source in kernel_sources,
            "explicit_exact_type_average_declared": mapping["gates"][
                "explicit_exact_type_average_declared"
            ],
            "exact_MaleCNS_type_body_set_available": anatomy["all_bodies_in_canonical_graph"],
            "soma_side_available": anatomy["soma_side_complete"],
            "complete_columnar_retinotopy_available": anatomy["columnar_retinotopy_available"],
        }
        rows[source] = {
            "mapping_mode": mapping["mapping_mode"],
            "recording_level_body_assignment": mapping["recording_level_body_assignment"],
            "MaleCNS_body_count": anatomy["body_count"],
            "MaleCNS_body_ids_sha256": anatomy["body_ids_sha256"],
            "gates": gates,
            "mapping_scope_complete": all(gates.values()),
        }
    four_tm_complete = all(rows[source]["mapping_scope_complete"] for source in kernel_sources)
    all_five_complete = all(rows[source]["mapping_scope_complete"] for source in required)
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_mappings": rows,
        "Tm1_Tm2_Tm4_Tm9_exact_type_average_mapping_complete": four_tm_complete,
        "T5_all_five_source_mapping_complete": all_five_complete,
        "CT1_mapping_boundary": {
            "body_ids": ct1_body_ids,
            "one_body_per_side": malecns["CT1"]["one_body_per_side"],
            "per_synapse_Lo1_columnar_retinotopy_available": ct1_columnar[
                "CT1_per_synapse_Lo1_columnar_retinotopy_available"
            ],
            "Lo1_column_count_by_body": ct1_counts,
            "complete_official_LO_column_coverage": ct1_columnar[
                "CT1_complete_official_LO_column_coverage"
            ],
            "single_body_column_coordinate_available": malecns["CT1"][
                "single_body_column_coordinate_available"
            ],
        },
        "stable_source_or_target_cell_to_MaleCNS_mapping": all_five_complete,
        "authorize_source_dynamics_transfer": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "four_Tm_exact_type_average_mappings_exist_but_CT1_allowed_payload_"
            "and_complete_columnar_coverage_are_missing"
        ),
        "boundary": config["boundary"],
    }
