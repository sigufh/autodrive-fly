import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_timebase_audit import (
    CONFIG,
    delay_mapping,
    leak_time_constants,
    stimulus_time_candidates,
)

ROOT = Path(__file__).parents[1]


def test_two_and_three_substep_delays_cannot_both_equal_160_ms() -> None:
    report = delay_mapping(
        {
            "centre_fast_excitatory": 0,
            "proximal_delayed_on_inhibitory": 2,
            "distal_delayed_off_inhibitory": 3,
        },
        160.0,
    )
    assert report["single_dt_matches_all_positive_delays"] is False
    assert (
        report["exact_single_branch_candidates"]["proximal_delayed_on_inhibitory"][
            "milliseconds_per_substep"
        ]
        == 80.0
    )
    assert np.isclose(
        report["exact_single_branch_candidates"]["distal_delayed_off_inhibitory"][
            "milliseconds_per_substep"
        ],
        160 / 3,
    )
    assert report["least_squares_rmse_milliseconds"] > 30
    assert report["paper_signed_offsets_milliseconds"] == {
        "pd": {"distal_Mi9": -160.0, "proximal_Mi4_C3": 160.0},
        "nd": {"distal_Mi9": 160.0, "proximal_Mi4_C3": -160.0},
    }
    assert report["current_signed_lags_substeps"] == {
        "all_directions": {"distal_Mi9": 3, "proximal_Mi4_C3": 2}
    }
    assert report["direction_dependent_sign_reversal_implemented"] is False
    assert report["positive_dt_can_match_paper_signed_offsets"] is False


def test_borrowed_angular_scale_exposes_inconsistent_stimulus_frame_durations() -> None:
    config = yaml.safe_load((ROOT / CONFIG).read_text())
    report = stimulus_time_candidates(config)
    assert report["borrowed_scale_is_valid_for_v7"] is False
    assert report["single_frame_duration_from_borrowed_scale"] is False
    candidates = report["candidates"]
    assert candidates["horizontal_edge"]["pixel_centres"][0] == 2
    assert candidates["horizontal_edge"]["pixel_centres"][-1] == 45
    assert candidates["vertical_edge"]["pixel_centres"][0] == 2
    assert candidates["vertical_edge"]["pixel_centres"][-1] == 21
    assert candidates["phase_grating"]["pixels_per_frame"] == 1
    assert (
        len({round(value["implied_milliseconds_per_frame"], 8) for value in candidates.values()})
        == 3
    )


def test_discrete_leaks_have_substep_but_not_physical_time_constants() -> None:
    report = leak_time_constants({"instant": 1.0, "slow": 0.1, "half": 0.5})
    assert report["instant"]["effective_time_constant_substeps"] == 0
    assert np.isclose(report["slow"]["effective_time_constant_substeps"], -1 / np.log(0.9))
    assert all(value["effective_time_constant_milliseconds"] is None for value in report.values())


def test_saved_timebase_audit_is_hash_bound_and_non_advancing() -> None:
    report = json.loads((ROOT / "artifacts/v7-timebase-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    current = report["current_v7_time_contract"]
    assert current["brain_substeps_per_frame"] == 4
    assert current["physical_frame_duration_defined"] is False
    assert current["physical_substep_duration_defined"] is False
    assert current["milliseconds_per_substep"] is None
    assert report["branched_delay_semantics"]["delays_substeps"] == {
        "centre_fast_excitatory": 0,
        "proximal_delayed_on_inhibitory": 2,
        "distal_delayed_off_inhibitory": 3,
    }
    conductance = report["conductance_candidate_timing"]
    assert conductance["uses_updated_same_substep_source_state"] is True
    assert conductance["uses_branched_delay_substeps"] is False
    assert conductance["paper_fixed_160_ms_shift_implemented"] is False
    assert report["identifiability"]["physical_timebase_identified"] is False
    assert report["identifiability"]["signed_pd_nd_sequence_matches_paper"] is False
    assert report["advance_to_time_calibrated_dynamics"] is False
    assert report["advance_to_parameter_fit"] is False
    assert report["advance_to_central_complex"] is False
    assert all(
        value["effective_time_constant_milliseconds"] is None
        for value in report["typed_leak_time_constants"].values()
    )
