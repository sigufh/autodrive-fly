import hashlib
import json
from pathlib import Path

import yaml

from fly_emotion.driving.v7_visual_target_input_audit import CONFIG

ROOT = Path(__file__).parents[1]


def test_visual_target_input_contract_does_not_infer_function_or_change_signs() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    assert config["t5_groups"]["fast_candidate_sources"] == ["Tm1", "Tm2", "Tm4"]
    assert config["t5_groups"]["delayed_candidate_sources"] == ["Tm9", "CT1"]
    assert config["looming_groups"]["target_types"] == ["LPLC1", "LPLC2", "LC4"]
    assert config["coverage_rules"]["looming_requires_any_T4_and_any_T5"] is False
    boundary = config["boundary"]
    assert boundary["anatomy_only"] is True
    assert boundary["infer_function_from_connectivity"] is False
    assert boundary["infer_inhibition_from_cell_type"] is False
    assert boundary["change_transmitter_sign"] is False
    assert boundary["remove_uncovered_cells"] is False


def test_saved_visual_target_audit_preserves_cells_and_known_structural_gaps() -> None:
    report = json.loads((ROOT / "artifacts/v7-visual-target-input-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    findings = report["structural_findings"]
    assert findings["T5_target_count"] == 6719
    assert findings["T5_targets_with_every_fast_and_any_delayed_source"] == 6717
    assert findings["T5_targets_with_all_five_candidate_sources"] == 6712
    assert findings["looming_target_count"] == 445
    assert findings["looming_targets_with_any_T4_and_any_T5"] == 319
    assert findings["LC4_requires_direct_T4_input"] is False
    assert len(report["T5"]["targets"]) == 6719
    assert len(report["looming_targets"]["targets"]) == 445
    assert report["T5"]["source_types"]["CT1"]["consensus_neurotransmitter_cells"] == {"gaba": 2}
    assert report["T5"]["source_types"]["Tm9"]["consensus_neurotransmitter_cells"] == {
        "acetylcholine": 1771
    }
    assert report["looming_targets"]["population_summary"]["LC4_R"]["coverage_counts"] == {
        "any_T4_input_present": 0,
        "any_T5_input_present": 55,
    }
    assert report["advance_to_model_change"] is False
    assert report["advance_to_validation"] is False
    assert report["advance_to_central_complex"] is False
