"""Bound candidate claims for independent Mi4/C3 whole-cell voltage data."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-mi4-c3-whole-cell-candidate-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_mi4_c3_whole_cell_candidate_audit.py"
)


def _verify_reference_document(root: Path, spec: dict) -> dict:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError("Groschner reference paper size changed")
    digest = _sha256(path)
    if digest != spec["sha256"]:
        raise ValueError("Groschner reference paper SHA-256 changed")
    reader = PdfReader(path, strict=False)
    text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in reader.pages
    )
    required_phrases = (
        "record the membrane potentials of direction-selective T4 neurons",
        "Aligned membrane voltage",
        "Mi9 Tm3 Mi1 Mi4 C3",
        "Data are available at the Edmond Open Research Data Repository",
    )
    if any(phrase not in text for phrase in required_phrases):
        raise ValueError("Groschner source-neuron voltage evidence text changed")
    return {**spec, "actual_sha256": digest, "pages": len(reader.pages)}


def evaluate_v7_mi4_c3_whole_cell_candidate_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    required_sources = set(config["required_sources"])
    if required_sources != {"Mi4", "C3"}:
        raise ValueError("Mi4/C3 required source set changed")
    document = _verify_reference_document(root, config["reference_document"])
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }

    groschner = evidence["groschner_T4_identity"]
    if not required_sources.issubset(groschner["source_summary"]):
        raise ValueError("Groschner Mi4/C3 source coverage changed")
    if not groschner["transfer_gates"][
        "all_four_T4_source_types_have_millivolt_traces"
    ]:
        raise ValueError("Groschner source voltage unit boundary changed")

    strother = evidence["strother_Mi4"]["source_evidence"]["Mi4"]
    strother_indexes = evidence["strother_Mi4_public_indexes"]
    if strother["measurement"]["response_unit"] != "deltaF_over_F":
        raise ValueError("Strother Mi4 modality changed")
    if strother["local_numeric_payload_verified"]:
        raise ValueError("Strother Mi4 numeric-payload boundary changed")
    if strother_indexes[
        "local_numeric_Mi4_trace_payload_found_in_successful_indexes"
    ]:
        raise ValueError("Strother public indexes gained an unreviewed payload")
    if strother_indexes["global_absence_claimed"]:
        raise ValueError("Strother public-index scope boundary changed")

    henning = evidence["henning_C3"]
    if henning["C3_dataset"]["STRF_units"] != "stimulus_response_correlation":
        raise ValueError("Henning C3 response-unit boundary changed")
    if not henning["C3_STRF_numerical_data_verified"]:
        raise ValueError("Henning C3 numeric payload is no longer verified")

    kohn = evidence["kohn_portes_T5"]
    kohn_sources = set(
        kohn["T5_source_contract"]["sources_with_local_numeric_membrane_voltage"]
    )
    if kohn_sources != {"Tm1", "Tm2", "Tm4", "Tm9"}:
        raise ValueError("Kohn-Portes whole-cell source coverage changed")
    if kohn_sources & required_sources:
        raise ValueError("Kohn-Portes now appears to cover Mi4 or C3")

    gruntman = evidence["gruntman_T4_T5"]["whole_cell_candidate"]
    if not gruntman["classification"]["supports_absolute_voltage_calibration"]:
        raise ValueError("Gruntman whole-cell modality boundary changed")
    if not evidence["gruntman_T4_T5"]["interface_status"][
        "processed_baseline_subtracted_T5_voltage_files_verified"
    ]:
        raise ValueError("Gruntman local processed voltage boundary changed")

    borst = evidence["borst_parameterized"]
    if borst["target_provenance"]["measured_membrane_voltage"]:
        raise ValueError("Borst target now appears to be measured membrane voltage")
    if "Mi4" not in borst["v7_source_coverage"]["covered_sources"]:
        raise ValueError("Borst Mi4 coverage changed")
    if "C3" not in borst["v7_source_coverage"]["missing_sources"]:
        raise ValueError("Borst C3 exclusion changed")

    molina_obando = evidence["molina_obando_source_data"]
    if not molina_obando["all_attachments_describe_mean_plus_minus_sem_tables"]:
        raise ValueError("Molina-Obando attachment summary-table boundary changed")
    if molina_obando["Mi4_or_C3_attachment_payload_found"]:
        raise ValueError("Molina-Obando attachments now appear to cover Mi4 or C3")
    if molina_obando["experimental_membrane_voltage_payload_found"]:
        raise ValueError("Molina-Obando attachments now appear to contain voltage")

    gonzalez_suarez = evidence["gonzalez_suarez_Mi4"]
    if not gonzalez_suarez["repository_evidence"]["Mi4_type_average_filter_available"]:
        raise ValueError("Gonzalez-Suarez Mi4 filter availability changed")
    if gonzalez_suarez["repository_evidence"]["C3_filter_available"]:
        raise ValueError("Gonzalez-Suarez C3 coverage changed")
    if gonzalez_suarez["independent_Mi4_experimental_membrane_voltage_available"]:
        raise ValueError("Gonzalez-Suarez Mi4 voltage boundary changed")

    yuan = evidence["yuan_C3"]
    if not yuan["C3_intervention_candidate_verified"]:
        raise ValueError("Yuan C3 intervention evidence changed")
    if yuan["C3_direct_recording_candidate_verified"]:
        raise ValueError("Yuan C3 direct-recording boundary changed")

    citation_graph = evidence["c3_citation_graph"]
    pang = citation_graph["high_relevance_candidate"]
    if pang["directly_recorded_neuron_types"] != ["L1", "L2"]:
        raise ValueError("Pang directly recorded neuron types changed")
    if pang["C3_direct_recording_found"]:
        raise ValueError("Pang now appears to contain direct C3 recording")

    hao = evidence["hao_ASAP7y_unresolved"]
    if not hao["verified_scope"]["Drosophila_in_vivo_voltage_imaging"]:
        raise ValueError("Hao ASAP7y Drosophila voltage scope changed")
    if hao["cell_type_resolution"]["candidate_classification"] != (
        "unresolved_high_value_candidate"
    ):
        raise ValueError("Hao ASAP7y candidate classification changed")
    if hao["authorize_Mi4_C3_candidate_classification"]:
        raise ValueError("Hao ASAP7y candidate was classified without cell types")

    stable_contrast = evidence["stable_contrast_scope"]
    if not stable_contrast["Mi4_C3_files_are_connectome_proofreading_only"]:
        raise ValueError("stable-contrast Mi4/C3 file scope changed")
    if stable_contrast["Mi4_direct_physiology_found"] or stable_contrast[
        "C3_direct_physiology_found"
    ]:
        raise ValueError("stable-contrast direct inhibitory-source coverage changed")

    tanaka = evidence["tanaka_Mi4_calcium"]
    if not tanaka["independent_Mi4_numerical_calcium_dynamics_verified"]:
        raise ValueError("Tanaka independent Mi4 calcium evidence changed")
    if tanaka["experimental_membrane_voltage"] or tanaka["C3_direct_recording_found"]:
        raise ValueError("Tanaka Mi4/C3 voltage boundary changed")

    candidates = config["candidates"]
    for name, candidate in candidates.items():
        measured = set(candidate["directly_measured_required_sources"])
        if not measured.issubset(required_sources):
            raise ValueError(f"candidate {name} contains a non-required measured source")
        expected_qualifies = bool(
            measured == required_sources
            and candidate["experimental_membrane_voltage"]
            and candidate["local_numeric_payload_verified"]
            and candidate["independent_biological_study_from_reference_cohort"]
        )
        candidate["qualifies_as_independent_Mi4_C3_numeric_membrane_voltage"] = (
            expected_qualifies
        )

    direct_voltage = [
        name
        for name, item in candidates.items()
        if set(item["directly_measured_required_sources"]) == required_sources
        and item["experimental_membrane_voltage"]
        and item["local_numeric_payload_verified"]
    ]
    independent_direct_voltage = [
        name
        for name in direct_voltage
        if candidates[name]["independent_biological_study_from_reference_cohort"]
    ]
    if direct_voltage != [config["reference_cohort"]]:
        raise ValueError("bounded direct Mi4/C3 voltage candidate set changed")

    gates = {
        "bounded_candidate_set_audited": True,
        "reference_Mi4_C3_numeric_membrane_voltage_verified": True,
        "independent_Mi4_C3_numeric_membrane_voltage_candidate_found": bool(
            independent_direct_voltage
        ),
        "independent_candidate_has_stable_biological_individual_ids": False,
        "independent_candidate_has_required_recording_fields": False,
        "independent_candidate_has_preregistered_split_roles": False,
        "independent_candidate_has_external_final_commitment": False,
    }
    authorized = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                config["reference_document"]["path"]: document["actual_sha256"],
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "reference_document": document,
        "candidate_matrix": candidates,
        "candidate_summary": {
            "audited_candidate_count": len(candidates),
            "direct_Mi4_C3_numeric_experimental_membrane_voltage_candidates": (
                direct_voltage
            ),
            "independent_direct_Mi4_C3_numeric_experimental_membrane_voltage_candidates": (
                independent_direct_voltage
            ),
            "claim_scope": "bounded_audited_candidate_set_not_global_nonexistence",
            "independent_Mi4_type_average_calcium_candidates": [
                "Gonzalez_Suarez_2022"
            ],
            "independent_Mi4_individual_calcium_candidates": ["Tanaka_2023"],
            "independent_C3_intervention_only_candidates": ["Yuan_2020"],
            "citation_graph_exclusion_candidates": ["Pang_2025"],
            "unresolved_high_value_candidates": ["Hao_2026_ASAP7y"],
            "unresolved_candidates_in_audited_candidate_count": False,
            "anatomy_only_named_source_candidates": ["Gur_2024"],
        },
        "transfer_gates": gates,
        "independent_Mi4_C3_voltage_transfer_authorized": authorized,
        "authorize_T4_source_dynamics_fit": authorized,
        "authorize_T4_functional_precheck": authorized,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if authorized
            else "only_reference_cohort_has_direct_Mi4_C3_numeric_membrane_voltage"
        ),
        "boundary": config["boundary"],
    }
