import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-t5-temporal-shuffle-input-energy-audit.json"


def test_shuffle_input_energy_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["condition_id"] == "S1-T01"
    assert report["protocol"]["stimulus_count"] == 120
    assert report["protocol"]["input_energy_metric"] == (
        "mean_absolute_value_over_time_and_units"
    )
    assert report["protocol"]["lamina_source_metric"] == (
        "positive_half_wave_of_signed_target_normalized_preactivation"
    )
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["target_activity_injection"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_shuffle_preserves_frames_but_not_transition_or_retinal_energy() -> None:
    report = json.loads(REPORT.read_text())
    assert report["frame_multiset_preserved_for_every_shuffle"] is True
    assert report["retinal_temporal_energy_exactly_preserved"] is False
    assert report["lamina_only_source_energy_exactly_preserved"] is False
    assert report["energy_matched_temporal_shuffle_control_verified"] is False
    assert report["temporal_shuffle_input_energy_amplified"] is True
    expected = {
        "1": (
            6.93426194129865,
            5.837713478478663,
            [4.908033023142721, 4.905535265160078, 4.902718804860586, 4.710199918312225],
            [0.4380513339747843, 0.4379028051340378, 0.43657969359191173, 0.45422753830080753],
        ),
        "4": (
            6.93426194129865,
            5.837713478478663,
            [5.158457721706764, 5.155179138220397, 5.148909292181183, 5.127327256460109],
            [0.43223462778410937, 0.43199250364279845, 0.43063522424125444, 0.4363820143500369],
        ),
    }
    for update, ratios in expected.items():
        result = report["by_brain_updates_per_frame"][update][
            "temporal_shuffle_to_ordered_energy_ratio"
        ]
        assert np.isclose(result["pixel_temporal_difference"], ratios[0])
        assert np.isclose(result["R1_R6_signed_frame_difference"], ratios[1])
        assert np.allclose(
            list(result["lamina_only_Tm_preactivation"].values()), ratios[2]
        )
        static = report["by_brain_updates_per_frame"][update][
            "static_sham_to_ordered_energy_ratio"
        ]
        assert np.isclose(static["pixel_temporal_difference"], 0.3333333416568422)
        assert np.isclose(static["R1_R6_signed_frame_difference"], 0.45652174898942627)
        assert np.allclose(
            list(static["lamina_only_Tm_preactivation"].values()), ratios[3]
        )


def test_input_energy_audit_does_not_rewrite_existing_output_results_or_gates() -> None:
    report = json.loads(REPORT.read_text())
    assert report["existing_output_ratios_recomputed"] is False
    assert report["existing_output_ratios_invalidated"] is False
    assert report["equal_energy_temporal_selectivity_interpretation_authorized"] is False
    assert report["authorize_new_energy_normalized_gate"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["direction_scoring_authorized"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
    assert report["advance_to_vehicle_experiments"] is False
