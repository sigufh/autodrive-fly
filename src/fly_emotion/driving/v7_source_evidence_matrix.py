"""Build a source-by-requirement matrix from frozen v7 evidence artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-source-evidence-matrix.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_source_evidence_matrix.py")


def evaluate_v7_source_evidence_matrix(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["contract"])
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    order = config["source_order"]
    expected_order = [
        *contract["required_families"]["T4"]["source_types"],
        *contract["required_families"]["T5"]["source_types"],
    ]
    if order != expected_order:
        raise ValueError("source evidence matrix order differs from contract")

    t4_voltage = evidence["t4_voltage"]
    t4_split = evidence["t4_individual_split"]
    edmond_retrieval = evidence["edmond_fig3_retrieval"]
    t4_fields = evidence["t4_recording_fields"]
    behnia_fast_sources = set(evidence["behnia_t4_fast"]["source_evidence"])
    behnia_availability = evidence["behnia_t4_fast_availability"]
    matulis_mi1 = evidence["matulis_mi1_voltage_availability"]
    inhibitory_external = evidence["t4_inhibitory_external"]["source_evidence"]
    mi4_c3_candidates = evidence["mi4_c3_whole_cell_candidates"]
    borst_2025 = evidence["borst_2025_temporal_filtering"]
    borst_2025_sources = set(borst_2025["v7_source_coverage"]["covered_sources"])
    pirogova = evidence["pirogova_source_calcium"]
    pirogova_sources = set(pirogova["source_payloads"])
    gou = evidence["gou_partial"]
    gou_sources = set(gou["source_evidence"])
    # The Gou archive was not downloaded or hash-verified. Its README describes
    # numerical arrays, but those arrays are deliberately not counted as local data.
    gou_payload_local = bool(
        gou["repositories"]["Dryad"]["payload_downloaded"]
        and gou["repositories"]["Dryad"]["payload_sha256_locally_verified"]
    )
    yang_sources = set(evidence["t5_voltage"]["source_evidence"])
    yang_external_index = evidence["t5_voltage"]["external_index_audit"]
    kohn_portes_voltage_sources = set(
        evidence["kohn_portes_t5_ephys"]["T5_source_contract"][
            "sources_with_local_numeric_membrane_voltage"
        ]
    )
    kohn_portes_identity = evidence["kohn_portes_identity_history"]
    t5_fields = evidence["t5_recording_fields"]
    t5_dynamic = {
        item["source_type"] for item in evidence["t5_calcium"]["verified_dynamic_blocks"].values()
    }
    t5_spatial = {
        item["source_type"] for item in evidence["t5_calcium"]["verified_spatial_blocks"].values()
    }
    braun = evidence["braun_t5_calcium"]
    braun_sources = set(braun["sources_with_local_temporal_calcium"])
    braun_identity_sources = set(braun["sources_with_stable_pseudonymous_fly_IDs"])
    arenz_t4 = set(evidence["arenz_t4"]["current_v7_source_contract"]["covered_by_Arenz"])
    arenz_t5 = set(evidence["arenz_t5"]["T5_source_contract"]["covered_sources"])
    mapping = evidence["malecns_mapping"]["source_mapping"]
    type_average_mapping = evidence["type_average_mapping"]["source_mappings"]
    tm9_coordinate = evidence["tm9_coordinate_identifiability"]

    c3_numerical = bool(evidence["c3_dynamics"]["C3_STRF_numerical_data_verified"])
    c3_ids = bool(evidence["c3_dynamics"]["C3_dataset"]["fly_count"] > 0)
    c3_external_disjoint = bool(
        evidence["c3_external_flash"]["transfer_gates"]["source_and_external_fly_ids_disjoint"]
    )
    c3_robustness = bool(
        evidence["c3_external_flash"]["transfer_gates"]["bootstrap_correlation_p05"]
    )
    ct1_type_average = bool(evidence["ct1_compartment"]["CT1_type_average_dynamics_verified"])
    ct1_lobula_phenotype = bool(
        evidence["ct1_compartment"]["T5_lobula_CT1_dynamic_phenotype_published"]
    )
    ct1_lobula_payload = bool(
        evidence["ct1_compartment"]["T5_lobula_CT1_numerical_payload_verified"]
    )
    ct1_simulated_voltage = bool(
        evidence["ct1_extreme_compartmentalization"]["model_evidence"]["output_is_simulated"]
    )
    ct1_pure_index = evidence["ct1_pure_data_index"]
    ct1_voltage_boundary = evidence["ct1_experimental_voltage_boundary"]

    rows = {}
    for source in order:
        family = config["source_family"][source]
        local_voltage = (
            family == "T4" and source in t4_voltage["source_summary"]
        ) or source in kohn_portes_voltage_sources
        published_voltage = source in yang_sources
        local_temporal_calcium = (
            source in t5_dynamic
            or source in pirogova_sources
            or source in braun_sources
            or (source == "C3" and c3_numerical)
        )
        local_spatial_calcium = source in t5_spatial
        local_type_average_deconvolved = source == "CT1" and ct1_type_average
        publisher_described_unverified = source in gou_sources and not gou_payload_local
        any_local_numerical_calcium = (
            local_temporal_calcium
            or local_spatial_calcium
            or local_type_average_deconvolved
            or (source in gou_sources and gou_payload_local)
        )
        published_calcium = (
            any_local_numerical_calcium
            or publisher_described_unverified
            or source == "CT1"
            and ct1_lobula_phenotype
        )
        physical_time = local_voltage or local_temporal_calcium or local_type_average_deconvolved
        individual_ids_any_numerical = (
            family == "T4"
            and evidence["t4_voltage"]["transfer_gates"][
                "stable_pseudonymous_biological_individual_ID_available"
            ]
        ) or (source == "C3" and c3_ids) or source in braun_identity_sources
        allowed_numerical = local_voltage
        individual_ids_on_allowed_payload = family == "T4" and individual_ids_any_numerical
        map_row = mapping[source]
        type_average_row = type_average_mapping[source]

        evidence_components = {
            "local_numerical_membrane_voltage": local_voltage,
            "published_optical_voltage_phenotype_only": published_voltage,
            "local_numerical_temporal_calcium_or_STRF": local_temporal_calcium,
            "local_numerical_spatial_calcium_only": local_spatial_calcium,
            "local_uncompartmented_type_average_deconvolved_calcium": (
                local_type_average_deconvolved
            ),
            "publisher_described_numerical_calcium_payload_not_locally_verified": (
                publisher_described_unverified
            ),
            "published_lobula_Lo1_dynamic_phenotype_only": (
                source == "CT1" and ct1_lobula_phenotype
            ),
            "local_lobula_Lo1_numerical_time_series": (source == "CT1" and ct1_lobula_payload),
            "simulated_compartmental_model_voltage_not_experimental": (
                source == "CT1" and ct1_simulated_voltage
            ),
            "audited_CT1_candidate_set_has_direct_experimental_voltage": (
                source == "CT1"
                and ct1_voltage_boundary["transfer_gates"][
                    "direct_CT1_experimental_voltage_phenotype_found"
                ]
            ),
            "audited_CT1_candidate_set_has_direct_Lo1_experimental_voltage": (
                source == "CT1"
                and ct1_voltage_boundary["transfer_gates"][
                    "direct_CT1_Lo1_experimental_voltage_found"
                ]
            ),
            "CT1_incremental_2025_2026_candidate_count": (
                len(
                    ct1_voltage_boundary["incremental_search_2025_2026"][
                        "relevant_candidates"
                    ]
                )
                if source == "CT1"
                else 0
            ),
            "CT1_incremental_2025_2026_direct_voltage_found": (
                source == "CT1"
                and bool(
                    ct1_voltage_boundary["incremental_search_2025_2026"][
                        "new_direct_CT1_experimental_voltage_candidates"
                    ]
                )
            ),
            "CT1_PuRe_official_archive_candidate_count": (
                ct1_pure_index["official_index"]["archive_candidate_count"]
                if source == "CT1"
                else 0
            ),
            "CT1_PuRe_archive_contents_verified": (
                source == "CT1" and ct1_pure_index["archive_contents_verified"]
            ),
            "CT1_PuRe_new_numerical_payload_verified": (
                source == "CT1"
                and ct1_pure_index["new_CT1_numerical_payload_verified"]
            ),
            "stable_biological_individual_ids_in_any_numerical_payload": (
                individual_ids_any_numerical
            ),
            "independent_external_cohort_with_disjoint_individual_ids": (
                source == "C3" and c3_external_disjoint
            ),
            "fixed_external_robustness_gate_passed": (source == "C3" and c3_robustness),
            "fixed_T4_individual_split_passed_for_ON_and_OFF": (
                family == "T4"
                and all(
                    t4_split["conditions"][condition]["sources"][source]["passed"]
                    for condition in ("on", "off")
                )
            ),
            "fixed_T4_split_sample_interval_milliseconds": (
                t4_split["protocol"]["sample_interval_milliseconds"]
                if family == "T4"
                else None
            ),
            "Edmond_1khz_payload_currently_verified": (
                family == "T4"
                and edmond_retrieval["gates"][
                    "all_four_local_payloads_hash_and_structure_verified"
                ]
            ),
            "Edmond_1khz_fixed_split_recompute_authorized": (
                family == "T4"
                and edmond_retrieval["gates"][
                    "full_resolution_fixed_split_recompute_authorized"
                ]
            ),
            "Edmond_complete_public_dataset_file_count": (
                t4_fields["complete_public_dataset_index"]["dataset_file_count"]
                if family == "T4"
                else 0
            ),
            "Edmond_Fig3_identity_or_metadata_sidecar_found": (
                family == "T4"
                and bool(
                    t4_fields["complete_public_dataset_index"][
                        "Fig3_identity_or_metadata_sidecars"
                    ]
                )
            ),
            "Edmond_workbook_recording_metadata_recovered": (
                family == "T4"
                and t4_fields["source_workbook_package"][
                    "recording_metadata_recovered_from_package"
                ]
            ),
            "Edmond_Fig1_Fig3_cohort_relation_identifiable": (
                family == "T4"
                and t4_fields["cross_figure_identity_boundary"][
                    "cohort_overlap_or_disjointness_identifiable"
                ]
            ),
            "Edmond_Fig3_edge_baseline_window_declared": (
                family == "T4"
                and t4_fields["paper_evidence"][
                    "Fig3_edge_recording_baseline_window_declared"
                ]
            ),
            "independent_whole_cell_voltage_phenotype_without_numeric_payload": (
                source in behnia_fast_sources
            ),
            "Behnia_2014_numeric_payload_found_in_audited_public_indexes": (
                source in behnia_fast_sources
                and behnia_availability[
                    "local_numeric_trace_payload_found_in_audited_indexes"
                ]
            ),
            "Matulis_2020_independent_Mi1_whole_cell_voltage_phenotype": (
                source == "Mi1"
                and matulis_mi1["transfer_gates"][
                    "independent_Mi1_experimental_voltage_phenotype_verified"
                ]
            ),
            "Matulis_2020_public_numeric_voltage_payload_available": (
                source == "Mi1"
                and matulis_mi1["transfer_gates"][
                    "public_numeric_voltage_payload_available"
                ]
            ),
            "Yang_2016_PMC_attachment_count": (
                yang_external_index["pmc_attachment_count"]
                if source in yang_sources
                else 0
            ),
            "Yang_2016_numeric_payload_found_in_successful_public_indexes": (
                source in yang_sources
                and (
                    bool(yang_external_index["crossref_data_links"])
                    or yang_external_index["datacite_related_count"] > 0
                    or yang_external_index["datacite_exact_title_count"] > 0
                    or yang_external_index["pmc_numeric_attachment_count"] > 0
                    or yang_external_index["github_exact_title_repository_count"] > 0
                    or yang_external_index["zenodo_exact_title_record_count"] > 0
                )
            ),
            "Yang_2016_Figshare_search_accessible": (
                source in yang_sources
                and yang_external_index["figshare_search_accessible"]
            ),
            "independent_inhibitory_source_physiology_published": (
                source in inhibitory_external
                and inhibitory_external[source]["independent_physiology_published"]
            ),
            "independent_inhibitory_source_allowed_unit": (
                source in inhibitory_external
                and inhibitory_external[source]["allowed_response_unit"]
            ),
            "bounded_candidate_set_direct_Mi4_C3_voltage_reference": (
                source in {"Mi4", "C3"}
                and mi4_c3_candidates["candidate_summary"][
                    "direct_Mi4_C3_numeric_experimental_membrane_voltage_candidates"
                ]
                == ["Groschner_2022"]
            ),
            "bounded_candidate_set_independent_Mi4_C3_voltage_found": (
                source in {"Mi4", "C3"}
                and mi4_c3_candidates["transfer_gates"][
                    "independent_Mi4_C3_numeric_membrane_voltage_candidate_found"
                ]
            ),
            "Borst_2025_parameterized_calcium_derived_target": (
                source in borst_2025_sources
            ),
            "Borst_2025_target_is_experimental_membrane_voltage": (
                source in borst_2025_sources
                and borst_2025["transfer_gates"]["experimental_membrane_voltage_payload"]
            ),
            "Pirogova_2023_local_numeric_calcium_time_series": (
                source in pirogova_sources
            ),
            "Pirogova_2023_allowed_membrane_voltage_unit": (
                source in pirogova_sources
                and pirogova["transfer_gates"]["allowed_membrane_voltage_response_unit"]
            ),
            "Pirogova_2023_biological_individual_id_available": (
                source in pirogova_sources
                and pirogova["source_payloads"][source][
                    "biological_individual_id_available"
                ]
            ),
            "Braun_2023_local_GCaMP7f_time_series": source in braun_sources,
            "Braun_2023_stable_source_local_fly_IDs": (
                source in braun_identity_sources
            ),
            "Braun_2023_complete_condition_grid": (
                source in set(braun["sources_with_complete_condition_grid"])
            ),
            "Braun_2023_allowed_membrane_voltage_unit": (
                source in set(braun["sources_with_allowed_response_unit"])
            ),
            "Braun_2023_baseline_window_declared": (
                source in braun_sources and braun["baseline_window_declared"]
            ),
            "Tm9_coordinate_identifiable_under_existing_rule": (
                source == "Tm9"
                and tm9_coordinate[
                    "Tm9_532266_coordinate_identifiable_under_existing_rule"
                ]
            ),
            "Tm9_post_hoc_coordinate_repair_authorized": (
                source == "Tm9"
                and tm9_coordinate["Tm9_532266_coordinate_repair_authorized"]
            ),
            "Tm9_latest_public_release_is_v1_0": (
                source == "Tm9"
                and tm9_coordinate["latest_public_release_check"][
                    "latest_public_release"
                ]
                == "male-cns:v1.0"
            ),
            "Tm9_current_annotation_object_matches_frozen_release": (
                source == "Tm9"
                and tm9_coordinate["latest_public_release_check"][
                    "current_object_matches_frozen_local_annotation"
                ]
            ),
            "Kohn_Portes_full_repository_recording_id_upper_bound": (
                kohn_portes_identity["source_capacity"][source][
                    "full_repository_unique_recording_id_upper_bound"
                ]
                if source in kohn_portes_voltage_sources
                else 0
            ),
            "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3": (
                source in kohn_portes_voltage_sources
                and kohn_portes_identity["source_capacity"][source][
                    "recording_id_upper_bound_meets_numeric_5_plus_3"
                ]
            ),
            "Kohn_Portes_biological_individual_5_plus_3_authorized": (
                source in kohn_portes_voltage_sources
                and kohn_portes_identity["source_capacity"][source][
                    "biological_individual_5_plus_3_split_authorized"
                ]
            ),
            "Kohn_Portes_record_specific_stimulus_provenance_complete": (
                family == "T5"
                and t5_fields["stimulus_provenance"][
                    "stimulus_provenance_contract_complete"
                ]
            ),
        }
        numerical_sources = []
        phenotype_sources = []
        publisher_described_sources = []
        if local_voltage:
            numerical_sources.append(
                "Kohn_Portes_whole_cell_voltage_flash_payload"
                if source in kohn_portes_voltage_sources
                else "T4_Fig3_verified_1khz_millivolt_array"
            )
        if source in gou_sources and gou_payload_local:
            numerical_sources.append("Gou_Dryad_processed_calcium")
        elif publisher_described_unverified:
            publisher_described_sources.append("Gou_Dryad_README_and_repository_metadata")
        if source in t5_dynamic:
            numerical_sources.append("Ramos_Traslosheros_temporal_calcium_workbook")
        if source in t5_spatial:
            numerical_sources.append("Ramos_Traslosheros_spatial_calcium_workbook")
        if source == "C3" and c3_numerical:
            numerical_sources.append("C3_STRF_calcium_repository")
        if local_type_average_deconvolved:
            numerical_sources.append("TimingModels_uncompartmented_type_average_CT1_filter")
        if source in arenz_t4 or source in arenz_t5:
            phenotype_sources.append("Arenz_2017_calcium_filter_parameters")
        if source in behnia_fast_sources:
            phenotype_sources.append("Behnia_2014_whole_cell_voltage_phenotype")
        if source == "Mi1" and matulis_mi1["transfer_gates"][
            "independent_Mi1_experimental_voltage_phenotype_verified"
        ]:
            phenotype_sources.append("Matulis_2020_Mi1_whole_cell_voltage_phenotype")
        if published_voltage:
            phenotype_sources.append("Yang_2016_optical_voltage_figure")
        if source == "CT1" and ct1_lobula_phenotype:
            phenotype_sources.append("TimingModels_lobula_Lo1_CT1_supplement_figure")
        if source in borst_2025_sources:
            phenotype_sources.append("Borst_2025_parameterized_calcium_derived_fit_target")
        if source in pirogova_sources:
            numerical_sources.append("Pirogova_2023_historical_GCaMP6f_time_series")
        if source in braun_sources:
            numerical_sources.append("Braun_2023_GCaMP7f_edge_time_series")

        missing = []
        if not allowed_numerical:
            missing.append("numerical_allowed_response_unit")
        if not individual_ids_on_allowed_payload:
            missing.append("stable_biological_individual_id_on_allowed_payload")
        if not map_row["columnar_retinotopy_available"]:
            missing.append("complete_columnar_retinotopy")
        if not type_average_row["gates"]["explicit_exact_type_average_declared"]:
            missing.append("external_recording_to_body_or_explicit_type_average")
        missing.extend(
            [
                "complete_stimulus_and_baseline_fields_on_allowed_payload",
                "training_validation_external_final_roles",
                "external_final_commitment",
            ]
        )
        if not physical_time:
            missing.append("local_numerical_physical_time_axis")
        row_gates = {
            "numerical_allowed_response_unit": allowed_numerical,
            "local_numerical_physical_time_axis": physical_time,
            "stable_biological_individual_id_on_allowed_payload": (
                individual_ids_on_allowed_payload
            ),
            "MaleCNS_exact_type_body_set": bool(map_row["all_bodies_in_canonical_graph"]),
            "soma_side": bool(map_row["soma_side_complete"]),
            "complete_columnar_retinotopy": bool(map_row["columnar_retinotopy_available"]),
            "complete_stimulus_and_baseline_fields_on_allowed_payload": (
                t4_fields["complete_stimulus_and_baseline_fields_on_allowed_payload"]
                if family == "T4"
                else t5_fields[
                    "all_five_T5_sources_have_complete_coexisting_recording_fields"
                ]
            ),
            "external_recording_to_body_or_explicit_type_average": bool(
                type_average_row["gates"]["explicit_exact_type_average_declared"]
            ),
            "training_validation_external_final_roles": False,
            "external_final_commitment": False,
        }
        rows[source] = {
            "family": family,
            "evidence_components": evidence_components,
            "numerical_membrane_voltage": local_voltage,
            "published_optical_voltage_phenotype": published_voltage,
            "local_numerical_calcium_or_deconvolved_calcium": (any_local_numerical_calcium),
            "publisher_described_unverified_numerical_payload": (publisher_described_unverified),
            "published_calcium_phenotype": published_calcium,
            "numerical_evidence_sources": numerical_sources,
            "publisher_described_sources": publisher_described_sources,
            "phenotype_evidence_sources": phenotype_sources,
            "MaleCNS_body_count": int(map_row["body_count"]),
            "MaleCNS_located_fraction": float(map_row["located_fraction"]),
            "gates": row_gates,
            "all_contract_gates_passed": all(row_gates.values()),
            "missing_requirements": missing,
        }

    family_summary = {}
    for family in ("T4", "T5"):
        sources = [source for source in order if config["source_family"][source] == family]
        family_summary[family] = {
            "sources": sources,
            "source_count": len(sources),
            "numerical_membrane_voltage_count": sum(
                rows[source]["numerical_membrane_voltage"] for source in sources
            ),
            "published_optical_voltage_phenotype_count": sum(
                rows[source]["published_optical_voltage_phenotype"] for source in sources
            ),
            "local_numerical_calcium_or_deconvolved_count": sum(
                rows[source]["local_numerical_calcium_or_deconvolved_calcium"] for source in sources
            ),
            "publisher_described_unverified_numerical_payload_count": sum(
                rows[source]["publisher_described_unverified_numerical_payload"]
                for source in sources
            ),
            "all_sources_contract_complete": all(
                rows[source]["all_contract_gates_passed"] for source in sources
            ),
        }
    matrix_ready = all(item["all_contract_gates_passed"] for item in rows.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "source_order": order,
        "matrix": rows,
        "family_summary": family_summary,
        "all_nine_sources_contract_complete": matrix_ready,
        "authorize_source_dynamics_fit": matrix_ready,
        "authorize_T4_T5_functional_precheck": matrix_ready,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "global_missing_requirements": sorted(
            {missing for row in rows.values() for missing in row["missing_requirements"]}
        ),
        "stop_reason": (
            None
            if matrix_ready
            else "no_source_currently_satisfies_all_unit_identity_mapping_and_split_gates"
        ),
        "boundary": config["boundary"],
    }
