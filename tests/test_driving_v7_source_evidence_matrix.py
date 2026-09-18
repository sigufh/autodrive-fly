import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-source-evidence-matrix.json"


def test_source_evidence_matrix_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["source_order"] == [
        "Mi1",
        "Tm3",
        "Mi4",
        "C3",
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "CT1",
    ]
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_T4_has_numerical_voltage_but_no_complete_source() -> None:
    report = json.loads(REPORT.read_text())
    assert report["family_summary"]["T4"]["numerical_membrane_voltage_count"] == 4
    assert report["family_summary"]["T4"]["all_sources_contract_complete"] is False
    for source in report["family_summary"]["T4"]["sources"]:
        assert report["matrix"][source]["numerical_membrane_voltage"] is True
        assert (
            report["matrix"][source]["gates"]["stable_biological_individual_id_on_allowed_payload"]
            is True
        )
        assert (
            report["matrix"][source]["gates"][
                "complete_stimulus_and_baseline_fields_on_allowed_payload"
            ]
            is False
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "fixed_T4_individual_split_passed_for_ON_and_OFF"
            ]
            is False
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "fixed_T4_split_sample_interval_milliseconds"
            ]
            == 1.0
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "Edmond_1khz_payload_currently_verified"
            ]
            is True
        )
        assert (
            report["matrix"][source]["evidence_components"][
                "Edmond_1khz_fixed_split_recompute_authorized"
            ]
            is True
        )
        row = report["matrix"][source]
        assert "T4_Fig3_verified_1khz_millivolt_array" in row["numerical_evidence_sources"]


def test_gou_readme_is_not_counted_as_local_numerical_payload() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Tm1", "Tm2"):
        row = report["matrix"][source]
        assert row["publisher_described_unverified_numerical_payload"] is True
        assert "Gou_Dryad_processed_calcium" not in row["numerical_evidence_sources"]
        assert row["publisher_described_sources"] == ["Gou_Dryad_README_and_repository_metadata"]
    assert (
        report["family_summary"]["T4"]["publisher_described_unverified_numerical_payload_count"]
        == 2
    )


def test_independent_T4_fast_phenotype_is_not_counted_as_numeric_payload() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3"):
        row = report["matrix"][source]
        assert (
            row["evidence_components"][
                "independent_whole_cell_voltage_phenotype_without_numeric_payload"
            ]
            is True
        )
        assert "Behnia_2014_whole_cell_voltage_phenotype" in row["phenotype_evidence_sources"]
    for source in ("Mi4", "C3"):
        assert (
            report["matrix"][source]["evidence_components"][
                "independent_whole_cell_voltage_phenotype_without_numeric_payload"
            ]
            is False
        )


def test_borst_2025_parameterized_target_is_not_counted_as_measured_voltage() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Mi4", "Tm1", "Tm2", "Tm4", "Tm9"):
        components = report["matrix"][source]["evidence_components"]
        assert components["Borst_2025_parameterized_calcium_derived_target"] is True
        assert components["Borst_2025_target_is_experimental_membrane_voltage"] is False
        assert "Borst_2025_parameterized_calcium_derived_fit_target" in (
            report["matrix"][source]["phenotype_evidence_sources"]
        )
    for source in ("C3", "CT1"):
        assert (
            report["matrix"][source]["evidence_components"][
                "Borst_2025_parameterized_calcium_derived_target"
            ]
            is False
        )


def test_pirogova_numeric_calcium_does_not_cross_unit_or_identity_boundaries() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Tm1", "Tm2"):
        row = report["matrix"][source]
        components = row["evidence_components"]
        assert components["Pirogova_2023_local_numeric_calcium_time_series"] is True
        assert components["Pirogova_2023_allowed_membrane_voltage_unit"] is False
        assert components["Pirogova_2023_biological_individual_id_available"] is False
        assert "Pirogova_2023_historical_GCaMP6f_time_series" in (
            row["numerical_evidence_sources"]
        )
    for source in ("Mi4", "C3", "Tm4", "Tm9", "CT1"):
        assert (
            report["matrix"][source]["evidence_components"][
                "Pirogova_2023_local_numeric_calcium_time_series"
            ]
            is False
        )


def test_exact_type_average_mapping_is_explicit_without_body_assignment() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi1", "Tm3", "Mi4", "C3", "Tm1", "Tm2", "Tm4", "Tm9"):
        assert (
            report["matrix"][source]["gates"]["external_recording_to_body_or_explicit_type_average"]
            is True
        )
    assert (
        report["matrix"]["CT1"]["gates"]["external_recording_to_body_or_explicit_type_average"]
        is False
    )
    tm9 = report["matrix"]["Tm9"]["evidence_components"]
    assert tm9["Tm9_coordinate_identifiable_under_existing_rule"] is False
    assert tm9["Tm9_post_hoc_coordinate_repair_authorized"] is False


def test_T5_modalities_are_not_combined_into_false_completeness() -> None:
    report = json.loads(REPORT.read_text())
    assert report["family_summary"]["T5"]["numerical_membrane_voltage_count"] == 4
    assert report["family_summary"]["T5"]["published_optical_voltage_phenotype_count"] == 2
    assert report["family_summary"]["T5"]["local_numerical_calcium_or_deconvolved_count"] == 5
    assert (
        report["matrix"]["Tm4"]["evidence_components"]["local_numerical_temporal_calcium_or_STRF"]
        is True
    )
    assert (
        report["matrix"]["Tm9"]["evidence_components"]["local_numerical_temporal_calcium_or_STRF"]
        is True
    )
    assert report["matrix"]["CT1"]["published_calcium_phenotype"] is True
    assert report["matrix"]["CT1"]["numerical_membrane_voltage"] is False
    for source in ("Tm1", "Tm2", "Tm4", "Tm9"):
        assert report["matrix"][source]["numerical_membrane_voltage"] is True
        assert report["matrix"][source]["numerical_evidence_sources"][0] == (
            "Kohn_Portes_whole_cell_voltage_flash_payload"
        )
    ct1 = report["matrix"]["CT1"]["evidence_components"]
    assert ct1["local_numerical_spatial_calcium_only"] is True
    assert ct1["local_uncompartmented_type_average_deconvolved_calcium"] is True
    assert ct1["published_lobula_Lo1_dynamic_phenotype_only"] is True
    assert ct1["local_lobula_Lo1_numerical_time_series"] is False
    assert ct1["simulated_compartmental_model_voltage_not_experimental"] is True
    assert ct1["audited_CT1_candidate_set_has_direct_experimental_voltage"] is False
    assert ct1["audited_CT1_candidate_set_has_direct_Lo1_experimental_voltage"] is False
    assert report["family_summary"]["T5"]["all_sources_contract_complete"] is False


def test_Kohn_Portes_recording_capacity_does_not_replace_biological_identity() -> None:
    report = json.loads(REPORT.read_text())
    expected = {"Tm1": 8, "Tm2": 7, "Tm4": 7, "Tm9": 13}
    for source, count in expected.items():
        components = report["matrix"][source]["evidence_components"]
        assert components["Kohn_Portes_full_repository_recording_id_upper_bound"] == count
        assert components["Kohn_Portes_biological_individual_5_plus_3_authorized"] is False
    assert report["matrix"]["Tm1"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is True
    assert report["matrix"]["Tm9"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is True
    assert report["matrix"]["Tm2"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is False
    assert report["matrix"]["Tm4"]["evidence_components"][
        "Kohn_Portes_recording_id_upper_bound_meets_5_plus_3"
    ] is False


def test_inhibitory_source_external_evidence_does_not_cross_modalities() -> None:
    report = json.loads(REPORT.read_text())
    for source in ("Mi4", "C3"):
        components = report["matrix"][source]["evidence_components"]
        assert components["independent_inhibitory_source_physiology_published"] is True
        assert components["independent_inhibitory_source_allowed_unit"] is False


def test_C3_identity_and_external_cohort_are_preserved_without_false_authorization() -> None:
    report = json.loads(REPORT.read_text())
    row = report["matrix"]["C3"]
    components = row["evidence_components"]
    assert components["local_numerical_temporal_calcium_or_STRF"] is True
    assert components["stable_biological_individual_ids_in_any_numerical_payload"] is True
    assert components["independent_external_cohort_with_disjoint_individual_ids"] is True
    assert components["fixed_external_robustness_gate_passed"] is False
    assert row["gates"]["stable_biological_individual_id_on_allowed_payload"] is True
    assert row["gates"]["complete_stimulus_and_baseline_fields_on_allowed_payload"] is False
    assert row["all_contract_gates_passed"] is False


def test_no_source_or_downstream_gate_is_authorized() -> None:
    report = json.loads(REPORT.read_text())
    assert all(not row["all_contract_gates_passed"] for row in report["matrix"].values())
    assert report["all_nine_sources_contract_complete"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"]["evidence_modalities_are_not_interchangeable"] is True
    assert report["boundary"]["publisher_descriptions_are_not_local_numerical_payloads"] is True
