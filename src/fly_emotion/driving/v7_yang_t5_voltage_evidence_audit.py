"""Audit Yang et al. Tm1/Tm2 optical-voltage evidence against the v7 contract."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-yang-t5-voltage-evidence-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_yang_t5_voltage_evidence_audit.py")


def evaluate_v7_yang_t5_voltage_evidence_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    supplement_reports = []
    combined_text = []
    for spec in config["supplements"]:
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]):
            raise ValueError("Yang supplement size mismatch")
        if _sha256(path) != spec["sha256"]:
            raise ValueError("Yang supplement SHA-256 mismatch")
        reader = PdfReader(path)
        if len(reader.pages) != int(spec["pages"]):
            raise ValueError("Yang supplement page count mismatch")
        attachment_count = len(reader.attachments or {})
        if attachment_count != 0:
            raise ValueError("Yang supplement unexpectedly embeds an attachment")
        text = " ".join(" ".join((page.extract_text() or "").split()) for page in reader.pages)
        combined_text.append(text)
        supplement_reports.append(
            {
                **spec,
                "actual_sha256": _sha256(path),
                "actual_pages": len(reader.pages),
                "embedded_attachment_count": attachment_count,
            }
        )
    text = " ".join(combined_text).replace("ﬂ", "fl").replace("ﬁ", "fi")
    required_fragments = (
        "constant frame rate of 38.9 Hz",
        "resampled our data from 38.9 Hz to 120 Hz",
        "Tm1 layer Lo1",
        "n = 30 cells, 3 flies",
        "Tm2 (red, n = 89 cells, 4 flies)",
        "Tm2 layer M2 (green; voltage: n = 89 cells, 6 flies",
        "25 ms light and dark flashes",
    )
    missing_fragments = [fragment for fragment in required_fragments if fragment not in text]
    if missing_fragments:
        raise ValueError(f"Yang supplement evidence changed: {missing_fragments}")
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required = contract["required_families"]["T5"]["source_types"]
    covered = [source for source in required if source in ("Tm1", "Tm2")]
    missing = [source for source in required if source not in covered]
    payload = config["payload"]
    gates = {
        "Tm1_lobula_Lo1_optical_voltage_phenotype_published": bool(
            config["evidence"]["Tm1"]["phenotype_reported"]
        ),
        "Tm2_medulla_M2_optical_voltage_phenotype_published": bool(
            config["evidence"]["Tm2"]["phenotype_reported"]
        ),
        "Tm2_fly_count_report_consistent": bool(
            config["evidence"]["Tm2"]["fly_count_report_consistent"]
        ),
        "every_T5_source_type_covered": not missing,
        "numerical_per_fly_time_series_available": bool(
            payload["numerical_per_fly_time_series_present"]
        ),
        "stable_recording_and_individual_IDs_available": bool(
            payload["stable_recording_unit_ids_present"]
            and payload["stable_biological_individual_ids_present"]
        ),
        "baseline_window_available": bool(payload["baseline_window_seconds_present"]),
        "MaleCNS_mapping_available": bool(payload["source_type_to_MaleCNS_mapping_present"]),
        "required_split_roles_available": bool(payload["required_split_roles_present"]),
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
            "supplements": supplement_reports,
            "pypdf_version": importlib.metadata.version("pypdf"),
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "measurement_protocol": {
            key: value for key, value in config["evidence"].items() if key not in ("Tm1", "Tm2")
        },
        "source_evidence": {
            "Tm1": config["evidence"]["Tm1"],
            "Tm2": config["evidence"]["Tm2"],
        },
        "T5_source_contract": {
            "required_sources": required,
            "sources_with_optical_voltage_phenotype": covered,
            "sources_without_optical_voltage_phenotype": missing,
            "phenotype_coverage_fraction": len(covered) / len(required),
        },
        "payload": payload,
        "transfer_gates": gates,
        "T5_partial_optical_voltage_phenotype_verified": True,
        "T5_numerical_voltage_payload_verified": bool(
            gates["numerical_per_fly_time_series_available"]
        ),
        "T5_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "Tm1_Tm2_voltage_phenotypes_lack_numerical_individual_payload_and_full_T5_coverage"
        ),
        "boundary": config["boundary"],
    }
