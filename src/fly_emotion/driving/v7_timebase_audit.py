from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7 import _moving_edge
from fly_emotion.driving.v7_branched import V7BranchedT4Probe
from fly_emotion.driving.v7_conductance import V7PublishedConductanceProbe
from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_phase_motion import PERIOD, PRE_FRAMES, phase_grating

CONFIG = Path("configs/driving-v7-timebase-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_timebase_audit.py")
V7_IMPLEMENTATION = Path("src/fly_emotion/driving/v7.py")
BRANCHED_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_branched.py")
CONDUCTANCE_IMPLEMENTATION = Path("src/fly_emotion/driving/v7_conductance.py")


def delay_mapping(delays: dict[str, int], target_milliseconds: float) -> dict:
    positive = {name: value for name, value in delays.items() if value > 0}
    exact_candidates = {
        name: {
            "milliseconds_per_substep": target_milliseconds / delay,
            "implied_delays_milliseconds": {
                other: other_delay * target_milliseconds / delay
                for other, other_delay in positive.items()
            },
        }
        for name, delay in positive.items()
    }
    values = np.asarray(list(positive.values()), dtype=float)
    least_squares_dt = float(target_milliseconds * values.sum() / np.sum(values**2))
    implied = {name: delay * least_squares_dt for name, delay in positive.items()}
    errors = {name: value - target_milliseconds for name, value in implied.items()}
    return {
        "target_milliseconds": target_milliseconds,
        "delays_substeps": delays,
        "exact_single_branch_candidates": exact_candidates,
        "single_dt_matches_all_positive_delays": len(set(positive.values())) == 1,
        "least_squares_milliseconds_per_substep": least_squares_dt,
        "least_squares_implied_delays_milliseconds": implied,
        "least_squares_errors_milliseconds": errors,
        "least_squares_rmse_milliseconds": float(
            np.sqrt(np.mean(np.square(list(errors.values()))))
        ),
        "paper_signed_offsets_milliseconds": {
            "pd": {"distal_Mi9": -target_milliseconds, "proximal_Mi4_C3": target_milliseconds},
            "nd": {"distal_Mi9": target_milliseconds, "proximal_Mi4_C3": -target_milliseconds},
        },
        "current_signed_lags_substeps": {
            "all_directions": {
                "distal_Mi9": positive.get("distal_delayed_off_inhibitory"),
                "proximal_Mi4_C3": positive.get("proximal_delayed_on_inhibitory"),
            }
        },
        "direction_dependent_sign_reversal_implemented": False,
        "positive_dt_can_match_paper_signed_offsets": False,
    }


def _edge_centres(frames: np.ndarray, *, axis: str, bright: bool) -> list[int]:
    foreground = 0.92 if bright else 0.08
    centres = []
    for frame in frames:
        line = frame[0] if axis == "x" else frame[:, 0]
        centres.append(int(np.flatnonzero(line == foreground)[-1]))
    return centres


def stimulus_time_candidates(config: dict) -> dict:
    assumptions = config["candidate_assumptions"]
    frames = int(assumptions["v7_frames"])
    substeps = int(assumptions["brain_substeps_per_frame"])
    paper = config["paper_reference"]
    pixel_degrees = float(paper["fig1_pixel_width_degrees"])
    speed = float(paper["edge_speed_degrees_per_second"])
    horizontal = _moving_edge(
        width=48, height=24, frames=frames, axis="x", direction=1, bright=True
    )
    vertical = _moving_edge(width=48, height=24, frames=frames, axis="y", direction=1, bright=True)
    centres = {
        "horizontal_edge": _edge_centres(horizontal, axis="x", bright=True),
        "vertical_edge": _edge_centres(vertical, axis="y", bright=True),
    }
    candidates = {}
    for name, values in centres.items():
        pixels_per_frame = (values[-1] - values[0]) / (len(values) - 1)
        frame_ms = pixels_per_frame * pixel_degrees / speed * 1000
        candidates[name] = {
            "pixel_centres": values,
            "mean_pixels_per_frame": pixels_per_frame,
            "borrowed_degrees_per_pixel": pixel_degrees,
            "borrowed_edge_speed_degrees_per_second": speed,
            "implied_milliseconds_per_frame": frame_ms,
            "implied_milliseconds_per_substep": frame_ms / substeps,
        }
    grating = phase_grating("right", 0).frames[PRE_FRAMES : PRE_FRAMES + PERIOD]
    if not np.array_equal(grating[1:], np.roll(grating[:-1], 1, axis=2)):
        raise ValueError("phase grating no longer translates one pixel per frame")
    grating_ms = float(assumptions["phase_grating_pixels_per_frame"]) * pixel_degrees / speed * 1000
    candidates["phase_grating"] = {
        "pixels_per_frame": float(assumptions["phase_grating_pixels_per_frame"]),
        "borrowed_degrees_per_pixel": pixel_degrees,
        "borrowed_edge_speed_degrees_per_second": speed,
        "implied_milliseconds_per_frame": grating_ms,
        "implied_milliseconds_per_substep": grating_ms / substeps,
    }
    durations = [value["implied_milliseconds_per_frame"] for value in candidates.values()]
    return {
        "candidates": candidates,
        "single_frame_duration_from_borrowed_scale": bool(np.allclose(durations, durations[0])),
        "borrowed_scale_is_valid_for_v7": False,
        "reason": (
            "Fig. 1 white-noise pixel width and Fig. 3 edge speed are not calibrated to the "
            "v7 split-eye camera; candidates expose inconsistency only"
        ),
    }


def leak_time_constants(leaks: dict[str, float]) -> dict:
    result = {}
    for name, value in leaks.items():
        alpha = float(value)
        result[name] = {
            "leak_per_substep": alpha,
            "effective_time_constant_substeps": (
                0.0 if alpha == 1.0 else float(-1.0 / np.log(1.0 - alpha))
            ),
            "effective_time_constant_milliseconds": None,
        }
    return result


def evaluate_v7_timebase_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract = yaml.safe_load((root / config["current_contract"]).read_text(encoding="utf-8"))
    branched = yaml.safe_load((root / config["branched_config"]).read_text(encoding="utf-8"))
    conductance = yaml.safe_load((root / config["conductance_config"]).read_text(encoding="utf-8"))
    ephys = json.loads((root / config["paper_evidence"]).read_text(encoding="utf-8"))
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("timebase audit must remain exploratory and non-advancing")
    paper = config["paper_reference"]
    if ephys["paper_protocol"]["repository_rate_hz"] != paper["repository_rate_hz"]:
        raise ValueError("paper sampling rate differs from frozen electrophysiology evidence")
    if ephys["paper_model_replay"]["direction_synthesis"]["shift_samples"] != 160:
        raise ValueError("paper shift differs from frozen electrophysiology evidence")
    visual = contract["controlled_vision"]
    if int(visual["brain_substeps_per_frame"]) != int(
        config["candidate_assumptions"]["brain_substeps_per_frame"]
    ):
        raise ValueError("v7 substep count differs from frozen timebase audit")
    if int(visual["frames_per_stimulus"]) != int(config["candidate_assumptions"]["v7_frames"]):
        raise ValueError("v7 stimulus frame count differs from frozen timebase audit")
    if int(branched["history_substeps"]) < max(
        config["candidate_assumptions"]["branched_delay_substeps"].values()
    ):
        raise ValueError("branched history cannot represent configured delays")
    if int(conductance["brain_substeps_per_frame"]) != int(visual["brain_substeps_per_frame"]):
        raise ValueError("conductance and v7 substep counts differ")
    delays = {name: int(value["delay_substeps"]) for name, value in branched["branches"].items()}
    expected_delays = {
        name: int(value)
        for name, value in config["candidate_assumptions"]["branched_delay_substeps"].items()
    }
    if delays != expected_delays:
        raise ValueError("branched delays differ from frozen timebase audit")
    v7_source = (root / V7_IMPLEMENTATION).read_text(encoding="utf-8")
    contract_text = (root / config["current_contract"]).read_text(encoding="utf-8")
    physical_tokens = ("frame_duration_ms", "milliseconds_per_substep", "solver_dt_ms")
    physical_time_defined = any(
        token in v7_source or token in contract_text for token in physical_tokens
    )
    branched_source = inspect.getsource(V7BranchedT4Probe._branch_activity)
    conductance_source = inspect.getsource(V7PublishedConductanceProbe._advance)
    if "history[min(delay - 1, len(history) - 1)]" not in branched_source:
        raise ValueError("branched delay semantics changed")
    conductance_uses_branch_delay = (
        "_branch_activity" in conductance_source or "delay_substeps" in conductance_source
    )
    source_leaks = visual["typed_visual_leak_v1"]
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                config["paper_evidence"]: _sha256(root / config["paper_evidence"]),
                config["current_contract"]: _sha256(root / config["current_contract"]),
                config["branched_config"]: _sha256(root / config["branched_config"]),
                config["conductance_config"]: _sha256(root / config["conductance_config"]),
                str(V7_IMPLEMENTATION): _sha256(root / V7_IMPLEMENTATION),
                str(BRANCHED_IMPLEMENTATION): _sha256(root / BRANCHED_IMPLEMENTATION),
                str(CONDUCTANCE_IMPLEMENTATION): _sha256(root / CONDUCTANCE_IMPLEMENTATION),
            },
            "parameter_fitting": False,
            "runtime_modified": False,
        },
        "paper_reference": paper,
        "current_v7_time_contract": {
            "frames_per_stimulus": int(visual["frames_per_stimulus"]),
            "brain_substeps_per_frame": int(visual["brain_substeps_per_frame"]),
            "physical_frame_duration_defined": physical_time_defined,
            "physical_substep_duration_defined": physical_time_defined,
            "milliseconds_per_substep": None,
            "stimulus_time_unit": "abstract frame index",
        },
        "branched_delay_semantics": {
            "delays_substeps": delays,
            "delay_zero_reads": "current pre-update state",
            "positive_delay_reads": "history[delay_substeps-1]",
            "history_length_substeps": int(branched["history_substeps"]),
            "physical_time_calibrated": False,
        },
        "paper_shift_mapping": delay_mapping(delays, float(paper["spatial_shift_milliseconds"])),
        "stimulus_time_candidates": stimulus_time_candidates(config),
        "typed_leak_time_constants": leak_time_constants(source_leaks),
        "conductance_candidate_timing": {
            "brain_substeps_per_frame": int(conductance["brain_substeps_per_frame"]),
            "uses_updated_same_substep_source_state": "_normalized_source_state(updated)"
            in conductance_source,
            "uses_branched_delay_substeps": conductance_uses_branch_delay,
            "history_retained_substeps": 1,
            "paper_fixed_160_ms_shift_implemented": False,
        },
        "identifiability": {
            "physical_timebase_identified": False,
            "single_dt_matches_branched_delays_and_paper_shift": False,
            "signed_pd_nd_sequence_matches_paper": False,
            "single_frame_duration_matches_current_stimulus_generators": False,
            "current_delay_values_are_ordering_hyperparameters": True,
            "reason": (
                "No frame duration is declared; 2 and 3 substeps cannot both equal the paper's "
                "160 ms shift; fixed positive lags cannot reverse between PD and ND; stimulus "
                "generators imply different durations under borrowed angular assumptions; and "
                "the conductance candidate bypasses branch delays."
            ),
        },
        "next_protocol_boundary": {
            **config["audit_boundary"],
            "required_design": [
                "separate stimulus sampling interval from neural solver step",
                "resample measured 1-kHz traces without changing their physical time axis",
                "fit only on declared T4 training cells or conditions",
                "freeze absolute-mV and PD/ND validation metrics before fitting",
                "reserve independent T4 validation and obtain separate T5 constraints",
            ],
        },
        "limitations": [
            "The borrowed 2.8125-degree pixel scale is only an inconsistency check.",
            "The paper's 160-ms spatial shift is not a synaptic time constant.",
            "Discrete leak coefficients lack millisecond constants until solver dt is defined.",
            "No candidate time mapping is selected by neural response or driving performance.",
        ],
        "advance_to_time_calibrated_dynamics": False,
        "advance_to_parameter_fit": False,
        "advance_to_central_complex": False,
    }
