import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-ct1-experimental-voltage-boundary-audit.json"


def test_CT1_voltage_boundary_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for item in report["candidate_documents"].values():
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == (
            item["actual_sha256"]
        )
    for item in report["candidate_documents"].values():
        if item["format"] == "pdf":
            assert item["embedded_attachment_names"] == []
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["candidate_summary"]["audited_candidate_count"] == 10
    assert report["candidate_summary"]["claim_scope"] == (
        "bounded_audited_candidate_set_not_global_nonexistence"
    )


def test_direct_CT1_candidates_are_calcium_not_voltage() -> None:
    report = json.loads(REPORT.read_text())
    assert report["candidate_summary"]["direct_CT1_measurement_candidates"] == [
        "Groschner_2022",
        "Meier_Borst_2019",
        "Ramos_Traslosheros_2021",
    ]
    assert report["candidate_matrix"]["Ramos_Traslosheros_2021"]["modality"] == (
        "GCaMP6f_calcium_imaging"
    )
    groschner = report["candidate_matrix"]["Groschner_2022"]
    assert groschner["CT1_GCaMP6f_genotype_present"] is True
    assert groschner["CT1_ArcLight_genotype_present"] is False
    assert groschner["ArcLight_cells"] == ["Mi1", "Tm3"]
    assert report["candidate_summary"]["direct_CT1_experimental_voltage_candidates"] == []
    assert report["candidate_summary"]["direct_CT1_Lo1_experimental_voltage_candidates"] == []


def test_other_cell_voltage_and_model_voltage_do_not_unlock_CT1() -> None:
    report = json.loads(REPORT.read_text())
    assert report["candidate_matrix"]["Yang_2016"]["voltage_cell_types"] == [
        "Tm1",
        "Tm2",
    ]
    assert report["candidate_matrix"]["Kohn_Portes_2021"]["voltage_cell_types"] == [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
    ]
    assert report["candidate_matrix"]["Lappalainen_2024"]["simulated_voltage_present"] is True
    assert report["transfer_gates"]["direct_CT1_experimental_voltage_phenotype_found"] is False
    assert report["transfer_gates"]["direct_CT1_public_numeric_voltage_payload_found"] is False
    assert report["CT1_experimental_voltage_transfer_authorized"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False


def test_2025_2026_incremental_candidates_are_not_CT1_voltage() -> None:
    report = json.loads(REPORT.read_text())
    incremental = report["incremental_search_2025_2026"]
    assert incremental["date_window"] == ["2025-01-01", "2026-09-19"]
    assert incremental["Europe_PMC_CT1_title_abstract_hit_count"] == 4
    assert incremental["Europe_PMC_complex_tangential_hit_count"] == 0
    assert incremental["new_direct_CT1_experimental_voltage_candidates"] == []
    matrix = report["candidate_matrix"]
    assert matrix["Samara_Borst_2025"]["modality"] == (
        "FlyWire_connectome_polyadic_synapse_structure"
    )
    assert matrix["Henning_2026"]["modality"] == (
        "C2_C3_two_photon_GCaMP_calcium_imaging"
    )
    assert matrix["Okuno_2026"]["modality"] == (
        "reused_whole_brain_calcium_plus_FlyWire_FlyEM_connectomes"
    )
    for name in incremental["relevant_candidates"]:
        assert matrix[name]["direct_CT1_experimental_membrane_voltage"] is False
