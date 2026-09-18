import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-c3-strf-source-dynamics-audit.json"


def test_c3_strf_audit_is_revision_and_hash_bound() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["archived_revision"] == (
        "745c6114b72a61dabc9389a1f024dbe22f679edf"
    )
    assert report["protocol"]["checked_out_revision"] == report["protocol"][
        "archived_revision"
    ]
    source = report["protocol"]["source_data"]
    assert source["bytes"] == 16_593_794
    assert source["git_blob_sha1"] == "7933bd47daa9aec542755470d8ed264a15791820"
    assert source["sha256"] == (
        "d35a64e1ee0daa1c0ec79f18153244a345981b87bca14d1e06fc34acc6615383"
    )
    assert {item["git_blob_sha1"] for item in report["protocol"]["source_code"]} == {
        "1e42a1dea6560775e8b1a5d4ab8a6255ec8e1be4",
        "da2db57ac78ebdc14ab778dc25638e3d742584f7",
        "40cb51ae2ab0f7d7a389053453a372ddfb2485e6",
    }
    for item in [source, *report["protocol"]["source_code"]]:
        payload = (ROOT / item["path"]).read_bytes()
        header = f"blob {len(payload)}\0".encode()
        assert (
            hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()
            == item["git_blob_sha1"]
        )
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_c3_author_analysis_is_numerically_reproduced() -> None:
    report = json.loads(REPORT.read_text())
    dataset = report["C3_dataset"]
    assert dataset["fly_count"] == 8
    assert dataset["time_bin_seconds"] == 0.05
    assert dataset["author_reproduction_by_axis"] == {"Az": True, "El": True}
    azimuth = dataset["axes"]["Az"]
    elevation = dataset["axes"]["El"]
    assert (azimuth["selected_ROIs"], azimuth["retained_ROIs"]) == (25, 25)
    assert (elevation["selected_ROIs"], elevation["retained_ROIs"]) == (10, 10)
    assert abs(azimuth["mean_ON_peak_seconds"] + 0.082) < 1e-12
    assert abs(elevation["mean_ON_peak_seconds"] + 0.085) < 1e-12
    assert azimuth["alternating_ROI_split_causal_kernel_correlation"] > 0.98
    assert elevation["alternating_ROI_split_causal_kernel_correlation"] > 0.96
    assert dataset["cross_axis_unweighted_causal_kernel_correlation"] > 0.95


def test_c3_evidence_does_not_authorize_cross_paper_filter_transfer() -> None:
    report = json.loads(REPORT.read_text())
    combined = report["combined_source_contract"]
    assert combined["required_source_types"] == ["Mi1", "Tm3", "Mi4", "C3"]
    assert combined["all_source_types_have_some_direct_temporal_evidence"] is True
    assert combined["all_source_types_share_one_transferable_parameterization"] is False
    gates = report["transfer_gates"]
    assert gates["published_C3_numerical_STRFs_available"] is True
    assert gates["author_selection_and_ON_peak_summary_reproduced"] is True
    assert gates["C3_membrane_voltage_or_validated_deconvolved_kernel_available"] is False
    assert gates["C3_analytic_filter_parameters_available"] is False
    assert gates["physical_v7_timebase_available"] is False
    assert gates["stable_recorded_cell_to_MaleCNS_mapping_available"] is False
    assert report["C3_source_filter_candidate_authorized"] is False
    assert report["advance_to_functional_precheck"] is False
    assert report["boundary"]["no_target_activity_injection"] is True
    assert report["boundary"]["T5_and_LPLC_remain_frozen"] is True
