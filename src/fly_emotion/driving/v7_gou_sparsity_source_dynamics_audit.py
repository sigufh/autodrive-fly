"""Audit Gou et al. sparsity data against the v7 source-dynamics contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-gou-sparsity-source-dynamics-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_gou_sparsity_source_dynamics_audit.py"
)


def evaluate_v7_gou_sparsity_source_dynamics_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required_by_family = {
        family: item["source_types"]
        for family, item in contract["required_families"].items()
    }
    evidence = config["source_evidence"]
    covered_by_family = {
        family: [source for source in sources if source in evidence]
        for family, sources in required_by_family.items()
    }
    missing_by_family = {
        family: [source for source in sources if source not in evidence]
        for family, sources in required_by_family.items()
    }
    required_count = sum(len(sources) for sources in required_by_family.values())
    covered_count = sum(len(sources) for sources in covered_by_family.values())
    source_unit_gates = {
        source: bool(item["response_unit_contract_satisfied"])
        for source, item in evidence.items()
    }
    fields = config["stimulus_and_recording"]
    transfer_gates = {
        "every_T4_source_type_measured": not missing_by_family["T4"],
        "every_T5_source_type_measured": not missing_by_family["T5"],
        "every_covered_source_has_allowed_response_unit": all(
            source_unit_gates.values()
        ),
        "processed_rows_linked_to_stable_subject_ids": bool(
            fields["processed_fly_rows_to_DANDI_subject_ids_verified"]
        ),
        "exact_sample_timestamps_and_baselines_verified": bool(
            fields["exact_sample_timestamps_verified_from_payload"]
            and fields["baseline_windows_verified_from_payload"]
        ),
        "source_type_to_MaleCNS_mapping_available": bool(
            fields["MaleCNS_mapping_present"]
        ),
        "required_split_roles_present": bool(
            fields["preregistered_training_validation_external_final_roles_present"]
        ),
        "payload_hash_locally_verified": bool(
            config["dryad"]["payload_sha256_locally_verified"]
        ),
    }
    transferable = all(transfer_gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
            },
            "paper": config["paper"],
            "parameter_fit": False,
            "external_payload_evaluated": False,
            "runtime_modified": False,
        },
        "repositories": {
            "Dryad": config["dryad"],
            "DANDI": config["dandi"],
        },
        "source_evidence": evidence,
        "stimulus_and_recording": fields,
        "required_sources_by_family": required_by_family,
        "covered_required_sources_by_family": covered_by_family,
        "missing_required_sources_by_family": missing_by_family,
        "covered_required_source_count": covered_count,
        "required_source_count": required_count,
        "coverage_fraction": covered_count / required_count,
        "source_response_unit_gates": source_unit_gates,
        "transfer_gates": transfer_gates,
        "partial_source_dynamics_evidence_present": covered_count > 0,
        "complete_external_source_dynamics_evidence": transferable,
        "authorize_source_dynamics_fit": transferable,
        "advance_to_T4_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [
            name for name, passed in transfer_gates.items() if not passed
        ],
        "stop_reason": (
            None
            if transferable
            else "partial_Mi1_Tm3_Tm1_Tm2_fluorescence_evidence_does_not_satisfy_unified_contract"
        ),
        "download_disposition": {
            "Dryad_archive_download_authorized": False,
            "DANDI_bulk_download_authorized": False,
            "reason": (
                "metadata establishes incomplete source and unit coverage; retain as a "
                "candidate for later source-specific calcium validation only"
            ),
        },
        "boundary": config["boundary"],
    }
