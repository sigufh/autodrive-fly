import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-arenz-source-dynamics-audit.json"


def test_arenz_source_dynamics_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    supplement = report["protocol"]["supplement"]
    assert supplement["bytes"] == 5_368_176
    assert supplement["sha256"] == (
        "492f7349d196dfd1630f29639f0ce98c163dedf12f6bf305e075b454666fc455"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_arenz_parameters_cover_three_current_sources_with_good_raw_fits() -> None:
    report = json.loads(REPORT.read_text())
    parameters = report["paper_model"]["parameters"]
    assert parameters["Mi1"]["class"] == "band_pass"
    assert parameters["Tm3"]["class"] == "band_pass"
    assert parameters["Mi4"]["class"] == "low_pass"
    assert parameters["Mi9"]["class"] == "low_pass"
    assert set(report["paper_model"]["raw_temporal_fit_gates"].values()) == {True}
    contract = report["current_v7_source_contract"]
    assert contract["covered_by_Arenz"] == ["Mi1", "Mi4", "Tm3"]
    assert contract["missing_from_Arenz"] == ["C3"]
    assert contract["Arenz_sources_outside_current_contract"] == ["Mi9"]
    assert contract["coverage_fraction"] == 0.75


def test_arenz_parameters_do_not_authorize_incomplete_physical_transfer() -> None:
    report = json.loads(REPORT.read_text())
    gates = report["transfer_gates"]
    assert gates["supplement_verified"] is True
    assert gates["all_Arenz_raw_temporal_fits_pass"] is True
    assert gates["every_current_source_type_covered"] is False
    assert gates["physical_v7_timebase_available"] is False
    assert gates["stable_recorded_cell_to_MaleCNS_mapping_available"] is False
    assert report["source_filter_candidate_authorized"] is False
    assert report["advance_to_functional_precheck"] is False
    assert report["boundary"]["C3_not_measured_in_Arenz_2017"] is True
    assert report["boundary"]["calcium_filters_are_not_membrane_voltage_kernels"] is True
