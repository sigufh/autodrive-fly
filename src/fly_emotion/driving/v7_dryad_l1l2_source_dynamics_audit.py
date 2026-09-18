"""Reject a public L1/L2 dataset as evidence for the v7 T4/T5 source contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-dryad-l1l2-source-dynamics-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_dryad_l1l2_source_dynamics_audit.py"
)


def evaluate_v7_dryad_l1l2_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required_by_family = {
        family: item["source_types"]
        for family, item in contract["required_families"].items()
    }
    required = {source for sources in required_by_family.values() for source in sources}
    measured = set(config["official_scope"]["measured_cell_types"])
    covered = sorted(required & measured)
    missing_by_family = {
        family: [source for source in sources if source not in measured]
        for family, sources in required_by_family.items()
    }
    small_files = config["small_files"]
    publisher_hashes_present = all(
        len(item["publisher_reported_sha256"]) == 64 for item in small_files.values()
    )
    local_hashes_verified = all(
        item["local_sha256_verified"] for item in small_files.values()
    )
    transfer_gates = {
        "every_T4_source_type_measured": not missing_by_family["T4"],
        "every_T5_source_type_measured": not missing_by_family["T5"],
        "source_type_to_MaleCNS_mapping_available": bool(
            config["official_scope"]["male_cns_mapping_present"]
        ),
        "local_payload_hashes_verified_before_use": local_hashes_verified,
    }
    suitable = all(transfer_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
            },
            "parameter_fit": False,
            "external_payload_evaluated": False,
            "runtime_modified": False,
        },
        "dataset": config["dataset"],
        "small_file_integrity": {
            "files": small_files,
            "publisher_reported_hashes_present": publisher_hashes_present,
            "local_payload_hashes_verified": local_hashes_verified,
            "integrity_claim_boundary": (
                "publisher_reported_SHA256_only; payload bytes were not obtained "
                "and hashes were not locally recomputed"
            ),
        },
        "official_scope": config["official_scope"],
        "required_sources_by_family": required_by_family,
        "covered_required_sources": covered,
        "covered_required_source_count": len(covered),
        "required_source_count": len(required),
        "missing_required_sources_by_family": missing_by_family,
        "transfer_gates": transfer_gates,
        "suitable_external_source_dynamics_evidence": suitable,
        "authorize_source_dynamics_fit": suitable,
        "advance_to_T4_T5_functional_precheck": suitable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "scientific_rejection_reason": (
            None
            if suitable
            else "measures_L1_L2_not_any_required_T4_T5_source_type_and_has_no_MaleCNS_mapping"
        ),
        "download_disposition": {
            "large_dataset_download_authorized": False,
            "bytes_avoided": int(config["dataset"]["total_bytes"]),
            "reason": "official_metadata_already_establishes_zero_required_source_coverage",
        },
        "boundary": config["boundary"],
    }
