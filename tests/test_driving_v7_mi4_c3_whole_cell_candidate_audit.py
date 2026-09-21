import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-mi4-c3-whole-cell-candidate-audit.json"


def test_candidate_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_only_reference_cohort_has_direct_Mi4_C3_numeric_voltage() -> None:
    report = json.loads(REPORT.read_text())
    summary = report["candidate_summary"]
    assert summary["audited_candidate_count"] == 9
    assert summary[
        "direct_Mi4_C3_numeric_experimental_membrane_voltage_candidates"
    ] == ["Groschner_2022"]
    assert summary[
        "independent_direct_Mi4_C3_numeric_experimental_membrane_voltage_candidates"
    ] == []
    assert summary["claim_scope"] == (
        "bounded_audited_candidate_set_not_global_nonexistence"
    )
    assert summary["independent_Mi4_type_average_calcium_candidates"] == [
        "Gonzalez_Suarez_2022"
    ]
    assert summary["independent_C3_intervention_only_candidates"] == ["Yuan_2020"]


def test_candidate_exclusions_preserve_target_and_modality_boundaries() -> None:
    matrix = json.loads(REPORT.read_text())["candidate_matrix"]
    assert matrix["Groschner_2022"][
        "independent_biological_study_from_reference_cohort"
    ] is False
    assert matrix["Strother_2018"]["response_unit"] == "deltaF_over_F"
    assert matrix["Henning_2025"]["response_unit"] == (
        "stimulus_response_correlation"
    )
    assert matrix["Kohn_Portes_2021"]["directly_measured_required_sources"] == []
    assert matrix["Gruntman_2021"]["target_neurons"] == ["T4", "T5"]
    assert matrix["Borst_2025"]["experimental_membrane_voltage"] is False
    molina_obando = matrix["Molina_Obando_2019"]
    assert molina_obando["target_neurons"] == ["Mi1", "Tm3"]
    assert molina_obando["directly_measured_required_sources"] == []
    assert molina_obando["experimental_membrane_voltage"] is False
    gonzalez_suarez = matrix["Gonzalez_Suarez_2022"]
    assert gonzalez_suarez["directly_measured_required_sources"] == ["Mi4"]
    assert gonzalez_suarez["response_unit"] == "normalized_type_average_filter"
    assert gonzalez_suarez["experimental_membrane_voltage"] is False
    yuan = matrix["Yuan_2020"]
    assert yuan["directly_measured_required_sources"] == []
    assert yuan["experimental_membrane_voltage"] is False
    assert yuan["local_numeric_payload_verified"] is False
    assert not any(
        item["qualifies_as_independent_Mi4_C3_numeric_membrane_voltage"]
        for item in matrix.values()
    )


def test_candidate_exclusion_keeps_all_downstream_gates_closed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["independent_Mi4_C3_voltage_transfer_authorized"] is False
    assert report["authorize_T4_source_dynamics_fit"] is False
    assert report["authorize_T4_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
    assert report["boundary"][
        "bounded_audited_candidate_set_not_global_nonexistence"
    ] is True
