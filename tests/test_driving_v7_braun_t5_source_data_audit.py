import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-braun-t5-source-data-audit.json"


def test_braun_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["dataset"]["manifest_file_count_verified"] is True
    assert report["dataset"]["all_selected_files_match_publisher_MD5"] is True
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_three_source_payloads_keep_fly_trial_time_and_stimulus_axes() -> None:
    report = json.loads(REPORT.read_text())
    assert report["sources_with_local_temporal_calcium"] == ["CT1", "Tm2", "Tm9"]
    assert report["sources_with_stable_pseudonymous_fly_IDs"] == ["CT1", "Tm2", "Tm9"]
    assert report["sources_with_complete_condition_grid"] == ["CT1", "Tm2", "Tm9"]
    assert {name: item["fly_count"] for name, item in report["payloads"].items()} == {
        "Tm2": 9,
        "Tm9": 11,
        "CT1": 9,
    }
    for item in report["payloads"].values():
        assert item["response_unit"] == "deltaF_over_F"
        assert item["indicator"] == "GCaMP7f"
        assert item["finite_fraction"] == 1.0
        assert item["full_condition_grid"] is True
        assert item["unique_values"]["name"] == ["stimuli.default.Edges"]
        assert item["unique_values"]["rotation [deg]"] == [0.0, 90.0, 180.0, 270.0]
        assert item["unique_values"]["trial"] == ["0", "1", "2"]


def test_calcium_identity_does_not_cross_into_voltage_or_mapping_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["sources_with_allowed_response_unit"] == []
    assert report["all_payload_fly_counts_match_paper_figure_5"] is False
    assert report["baseline_window_declared"] is False
    assert report["source_to_MaleCNS_mapping_available"] is False
    assert report["required_split_roles_available"] is False
    assert report["authorize_source_dynamics_fit"] is False
    assert report["authorize_T4_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
