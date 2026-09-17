import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-three-hop-source-coverage.json"


def test_three_hop_source_coverage_is_structural_and_non_advancing() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["path_length_edges"] == 3
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False
    assert report["results"]["balanced_1914"]["receptor_count"] == 1914
    assert report["results"]["all_mapped_3377"]["receptor_count"] == 3377


def test_independent_fast_delayed_paths_do_not_cover_every_population() -> None:
    report = json.loads(REPORT.read_text())
    balanced = report["results"]["balanced_1914"]["groups"]
    full = report["results"]["all_mapped_3377"]["groups"]
    for name in ("T4_center", "T4_proximal", "T4_distal", "T5_fast", "T5_delayed"):
        assert balanced[name]["all_populations_passed"] is False
        assert full[name]["all_populations_passed"] is False
    assert balanced["T4_all_known"]["all_populations_passed"] is True
    assert balanced["T5_all_known"]["all_populations_passed"] is True
    assert report["independent_fast_delayed_coverage_gate_passed"] is False
    assert report["authorize_scalar_reichardt"] is False
    assert report["advance_to_calibration"] is False
