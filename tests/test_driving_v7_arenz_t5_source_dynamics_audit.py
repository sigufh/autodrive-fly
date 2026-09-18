import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-arenz-t5-source-dynamics-audit.json"


def test_arenz_t5_audit_is_hash_bound_and_complete() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    contract = report["T5_source_contract"]
    assert contract["required_sources"] == ["Tm1", "Tm2", "Tm4", "Tm9"]
    assert set(contract["covered_sources"]) == {"Tm1", "Tm2", "Tm4", "Tm9"}
    assert contract["coverage_fraction"] == 1.0
    assert report["raw_T5_calcium_filter_contract_complete"] is True


def test_arenz_t5_deconvolved_tm9_and_transfer_fail() -> None:
    report = json.loads(REPORT.read_text())
    paper = report["paper_model"]
    assert paper["parameters"]["Tm9"]["deconvolved"]["R2"] == 0.273
    assert paper["deconvolved_temporal_fit_gates"]["Tm9"] is False
    assert report["deconvolved_T5_filter_contract_complete"] is False
    assert report["transfer_gates"]["physical_v7_timebase_available"] is False
    assert report["transfer_gates"][
        "stable_recorded_cell_to_MaleCNS_mapping_available"
    ] is False
    assert report["T5_source_filter_transfer_authorized"] is False
    assert report["advance_to_T5_functional_precheck"] is False
    assert report["boundary"]["Tm9_kept_separate_from_CT1"] is True
    assert report["boundary"]["no_T4_axis_mapping_reused"] is True
