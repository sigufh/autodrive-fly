"""Audit independent Mi4 and C3 evidence without mixing response modalities."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-inhibitory-source-external-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_inhibitory_source_external_audit.py")


def evaluate_v7_t4_inhibitory_source_external_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    mi4 = config["Mi4_paper"]
    source_path = root / mi4["source"]["path"]
    if source_path.stat().st_size != int(mi4["source"]["bytes"]):
        raise ValueError("Strother Mi4 HTML size mismatch")
    if _sha256(source_path) != mi4["source"]["sha256"]:
        raise ValueError("Strother Mi4 HTML SHA-256 mismatch")
    html = source_path.read_text(encoding="utf-8")
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    required_phrases = (
        "GCaMP6f for Mi1, Tm3, Mi4, and Mi9",
        "n = 5 for each genotype",
        "slow speed of the calcium indicator GCaMP6s",
        "unlikely to accurately capture the true kinetics of the Mi4 response",
        "All data, reagents, and code used in this manuscript will be provided upon request",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Strother Mi4 evidence text changed")
    hrefs = re.findall(r'href="([^"]+)"', html)
    numeric_extensions = (".xlsx", ".xls", ".csv", ".npy", ".npz", ".mat", ".zip")
    numeric_links = sorted({href for href in hrefs if href.lower().endswith(numeric_extensions)})

    c3_paths = {name: Path(path) for name, path in config["C3_evidence"].items()}
    c3 = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in c3_paths.items()
    }
    attachment_paths = {
        name: Path(path) for name, path in config["attachment_evidence"].items()
    }
    attachments = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in attachment_paths.items()
    }
    molina_obando = attachments["Molina_Obando_2019"]
    if molina_obando["attachment_count"] != 11:
        raise ValueError("Molina-Obando attachment count changed")
    if molina_obando["Mi4_or_C3_attachment_payload_found"]:
        raise ValueError("Molina-Obando attachments now appear to cover Mi4 or C3")
    additional_mi4_paths = {
        name: Path(path) for name, path in config["additional_Mi4_evidence"].items()
    }
    additional_mi4 = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in additional_mi4_paths.items()
    }
    gonzalez_suarez = additional_mi4["Gonzalez_Suarez_2022"]
    if not gonzalez_suarez["independent_Mi4_calcium_type_average_available"]:
        raise ValueError("Gonzalez-Suarez Mi4 calcium evidence changed")
    if gonzalez_suarez["independent_Mi4_experimental_membrane_voltage_available"]:
        raise ValueError("Gonzalez-Suarez Mi4 voltage boundary changed")
    c3_strf = c3["STRF"]
    c3_flash = c3["independent_flash"]
    allowed_units = contract["allowed_response_units"]
    mi4_measurement = mi4["measurement"]
    rows = {
        "Mi4": {
            "independent_physiology_published": True,
            "measurement": mi4_measurement,
            "public_numeric_payload_links_found": numeric_links,
            "local_numeric_payload_verified": False,
            "allowed_response_unit": mi4_measurement["response_unit"] in allowed_units,
            "independent_fixed_robustness_passed": False,
        },
        "C3": {
            "independent_physiology_published": True,
            "measurement": {
                "response_unit": c3_strf["C3_dataset"]["STRF_units"],
                "STRF_fly_count": c3_strf["C3_dataset"]["fly_count"],
                "external_flash_fly_count": c3_flash["cohorts"]["external_flash_fly_count"],
                "source_and_external_fly_ids_disjoint": c3_flash["transfer_gates"][
                    "source_and_external_fly_ids_disjoint"
                ],
                "mean_waveform_correlation": c3_flash["external_validation"][
                    "mean_waveform_correlation"
                ],
                "bootstrap_correlation_p05": c3_flash["external_validation"]["bootstrap"][
                    "correlation_p05"
                ],
            },
            "local_numeric_payload_verified": c3_strf["C3_STRF_numerical_data_verified"],
            "allowed_response_unit": c3_strf["C3_dataset"]["STRF_units"] in allowed_units,
            "independent_fixed_robustness_passed": c3_flash["transfer_gates"][
                "bootstrap_correlation_p05"
            ],
        },
    }
    complete = all(
        row["local_numeric_payload_verified"]
        and row["allowed_response_unit"]
        and row["independent_fixed_robustness_passed"]
        for row in rows.values()
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                **{str(path): _sha256(root / path) for path in c3_paths.values()},
                **{
                    str(path): _sha256(root / path)
                    for path in attachment_paths.values()
                },
                **{
                    str(path): _sha256(root / path)
                    for path in additional_mi4_paths.values()
                },
            },
            "Mi4_paper": {**mi4, "source_actual_sha256": _sha256(source_path)},
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_evidence": rows,
        "official_source_data_attachments": {
            "paper": molina_obando["paper"],
            "attachment_count": molina_obando["attachment_count"],
            "all_mean_plus_minus_sem_tables": molina_obando[
                "all_attachments_describe_mean_plus_minus_sem_tables"
            ],
            "Mi1_Tm3_GCaMP_summary_evidence_found": molina_obando[
                "Mi1_Tm3_GCaMP_summary_evidence_found"
            ],
            "Mi4_or_C3_attachment_payload_found": molina_obando[
                "Mi4_or_C3_attachment_payload_found"
            ],
            "individual_source_dynamics_payload_found": molina_obando[
                "individual_source_dynamics_payload_found"
            ],
            "experimental_membrane_voltage_payload_found": molina_obando[
                "experimental_membrane_voltage_payload_found"
            ],
        },
        "additional_Mi4_evidence": {
            "Gonzalez_Suarez_2022": {
                "paper": gonzalez_suarez["paper"],
                "Mi4_GCaMP6f_fly_count": gonzalez_suarez[
                    "paper_measurement_evidence"
                ]["Mi4_GCaMP6f_fly_count"],
                "Mi4_type_average_filter_available": gonzalez_suarez[
                    "repository_evidence"
                ]["Mi4_type_average_filter_available"],
                "filter_sample_interval_seconds": gonzalez_suarez[
                    "repository_evidence"
                ]["filter_sample_interval_seconds"],
                "individual_cell_axis_available": gonzalez_suarez[
                    "repository_evidence"
                ]["individual_cell_axis_available"],
                "Mi4_experimental_membrane_voltage_available": gonzalez_suarez[
                    "independent_Mi4_experimental_membrane_voltage_available"
                ],
                "C3_source_dynamics_available": gonzalez_suarez[
                    "independent_C3_source_dynamics_available"
                ],
            }
        },
        "both_inhibitory_sources_have_transferable_external_validation": complete,
        "authorize_T4_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            None
            if complete
            else "Mi4_public_evidence_is_calcium_only_and_C3_fixed_robustness_fails"
        ),
        "boundary": config["boundary"],
    }
