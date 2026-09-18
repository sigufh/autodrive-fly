"""Audit independent published Mi1/Tm3 voltage evidence and payload availability."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-behnia-t4-fast-source-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_behnia_t4_fast_source_audit.py")


def evaluate_v7_behnia_t4_fast_source_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    source = config["source"]
    source_path = root / source["path"]
    if source_path.stat().st_size != int(source["bytes"]):
        raise ValueError("Behnia 2014 HTML size mismatch")
    if _sha256(source_path) != source["sha256"]:
        raise ValueError("Behnia 2014 HTML SHA-256 mismatch")
    html = source_path.read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    required_phrases = (
        "whole-cell current-clamp recordings",
        "For Mi1 the average peak response time was 71 ms",
        "while it was 53 ms",
        "difference of 18 ms existed between the peak times",
        "Mi1 (N=7) and Tm3 (N=10)",
        "Mi1 (N=7) and Tm3 (N=11)",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Behnia 2014 evidence text changed")
    hrefs = re.findall(r'href="([^"]+)"', html)
    data_extensions = (".xlsx", ".xls", ".csv", ".npy", ".npz", ".zip", ".mat")
    numeric_links = sorted({href for href in hrefs if href.lower().endswith(data_extensions)})
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    reported = config["reported_evidence"]
    inventory = config["public_resource_inventory"]
    sources = {
        source_type: {
            **details,
            "numerical_membrane_voltage_phenotype_published": True,
            "local_numeric_payload_verified": False,
        }
        for source_type, details in reported["sources"].items()
    }
    gates = {
        "independent_Mi1_Tm3_whole_cell_phenotype_published": True,
        "independent_Mi1_later_than_Tm3_phenotype_published": (
            reported["Mi1_minus_Tm3_peak_latency_milliseconds"] > 0
        ),
        "all_four_T4_sources_covered": set(sources)
        == set(contract["required_families"]["T4"]["source_types"]),
        "local_numeric_trace_payload_verified": bool(
            inventory["local_numeric_trace_payload_present"] and numeric_links
        ),
        "stable_biological_individual_ids_available": bool(
            inventory["biological_individual_ids_present"]
        ),
        "source_to_MaleCNS_mapping_available": bool(inventory["source_to_MaleCNS_mapping_present"]),
        "required_split_roles_available": bool(inventory["preregistered_split_roles_present"]),
        "external_final_commitment_available": bool(inventory["external_final_commitment_present"]),
    }
    transferable = all(gates.values())
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
            "source": {**source, "actual_sha256": _sha256(source_path)},
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_evidence": sources,
        "reported_protocol": {key: value for key, value in reported.items() if key != "sources"},
        "public_resource_inventory": {
            **inventory,
            "numeric_payload_links_found": numeric_links,
        },
        "transfer_gates": gates,
        "independent_T4_source_dynamics_transfer_authorized": transferable,
        "authorize_T4_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "independent_Mi1_Tm3_phenotype_lacks_numeric_payload_and_Mi4_C3_coverage"
        ),
        "boundary": config["boundary"],
    }
