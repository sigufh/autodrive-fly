import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_t5_label_audit import CONFIG

ROOT = Path(__file__).parents[1]


def test_label_audit_contract_forbids_posthoc_direction_assignment() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    boundary = config["audit_boundary"]
    assert boundary["download_full_archive"] is False
    assert boundary["assign_PD_code_without_plotting_source"] is False
    assert boundary["infer_PD_from_response_magnitude"] is False
    assert boundary["fit_allowed"] is False
    assert boundary["change_visual_gate"] is False


def test_saved_label_audit_is_hash_bound_and_stays_non_advancing() -> None:
    report = json.loads((ROOT / "artifacts/v7-t5-label-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    dataset = report["figure4_dataset"]
    assert dataset["doi"].lower() == "10.25378/janelia.11328086.v1".lower()
    assert dataset["size_bytes"] == 1_391_804_749
    assert "cc-by-nc-4.0" in dataset["license_identifiers"]
    decision = report["bounded_download_decision"]
    assert decision["archive_exceeds_budget"] is True
    assert decision["archive_downloaded"] is False
    assert decision["file_manifest_retrieved"] is False
    assert report["label_status"]["direction_code_to_PD_ND_mapping_verified"] is False
    assert report["label_status"]["biological_PD_code_assigned"] is None
    assert report["advance_to_model_scoring"] is False
    assert report["advance_to_visual_gate"] is False
    assert report["advance_to_central_complex"] is False
