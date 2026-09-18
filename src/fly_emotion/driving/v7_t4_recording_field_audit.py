"""Audit Fig. 3 T4 source payload fields against the frozen external contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-t4-recording-field-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t4_recording_field_audit.py")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def evaluate_v7_t4_recording_field_audit(root: Path) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("the T4 field audit requires pypdf") from exc
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    evidence_paths = {
        name: Path(config[name])
        for name in ("contract", "retrieval_evidence", "identity_evidence", "split_evidence")
    }
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    contract = evidence["contract"]
    required_fields = contract["required_recording_fields"]
    if list(config["field_status"]) != required_fields:
        raise ValueError("T4 field audit does not preserve contract field order")
    retrieval = evidence["retrieval_evidence"]
    if not retrieval["gates"]["all_four_local_payloads_hash_and_structure_verified"]:
        raise ValueError("T4 field audit requires four verified 1-kHz payloads")
    if not retrieval["gates"]["one_khz_cell_order_identity_verified"]:
        raise ValueError("T4 field audit requires verified array-to-workbook ordinals")
    identity = evidence["identity_evidence"]
    if not identity["transfer_gates"][
        "stable_pseudonymous_biological_individual_ID_available"
    ]:
        raise ValueError("T4 field audit requires paper-backed biological identity")
    split = evidence["split_evidence"]
    if split["protocol"]["sample_interval_milliseconds"] != 1.0:
        raise ValueError("T4 field audit requires the full-resolution split")

    paper_spec = config["paper_pdf"]
    paper_path = root / paper_spec["path"]
    paper = PdfReader(paper_path)
    if (
        paper_path.stat().st_size != int(paper_spec["bytes"])
        or _sha256(paper_path) != paper_spec["sha256"]
        or len(paper.pages) != int(paper_spec["pages"])
    ):
        raise ValueError("T4 paper payload changed")
    paper_text = " ".join(
        " ".join((page.extract_text() or "").replace("- ", "").split())
        for page in paper.pages
    )
    required_paper_phrases = (
        "Bright ON and dark OFF edges travelling at a velocity of 30° s−1",
        "number of cells, each of which was recorded in a different animal",
        "The responses of individual neurons of one type were temporally aligned",
    )
    if not all(phrase in paper_text for phrase in required_paper_phrases):
        raise ValueError("T4 paper stimulus or identity statement changed")

    notebook_spec = config["notebook"]
    notebook_path = root / notebook_spec["path"]
    if (
        notebook_path.stat().st_size != int(notebook_spec["bytes"])
        or _sha256(notebook_path) != notebook_spec["sha256"]
    ):
        raise ValueError("T4 Fig. 3 notebook changed")
    notebook_source = _notebook_source(notebook_path)
    required_notebook_snippets = (
        "fs = 1000",
        "stims = ['on', 'off']",
        "dirs = ['pd', 'nd']",
        "shift = int(4.8*fs/30)",
        "Tm3a = np.load('fig3_Tm3.npy')",
        "cells = [Mi9a, Tm3a, Mi1a, Mi4a, C3a]",
    )
    if not all(snippet in notebook_source for snippet in required_notebook_snippets):
        raise ValueError("T4 Fig. 3 notebook field semantics changed")

    led_spec = config["led_array"]
    led_path = root / led_spec["path"]
    if (
        led_path.stat().st_size != int(led_spec["bytes"])
        or _md5(led_path) != led_spec["md5"]
        or _sha256(led_path) != led_spec["sha256"]
    ):
        raise ValueError("T4 Fig. 3 LED payload changed")
    led = np.load(led_path, allow_pickle=False)
    if list(led.shape) != led_spec["shape"] or str(led.dtype) != led_spec["dtype"]:
        raise ValueError("T4 Fig. 3 LED structure changed")
    crossings = {}
    for stimulus, name in enumerate(("on", "off")):
        for direction, direction_name in enumerate(("pd", "nd")):
            values = led[stimulus, direction]
            indices = np.flatnonzero(values >= 0.5 if name == "on" else values < 0.5)
            crossings[f"{name}_{direction_name}"] = int(indices[0])

    field_status = config["field_status"]
    available = [name for name, item in field_status.items() if item["available"]]
    missing = [name for name, item in field_status.items() if not item["available"]]
    complete_stimulus_and_baseline = all(
        field_status[name]["available"]
        for name in (
            "stimulus_id",
            "stimulus_family",
            "stimulus_polarity",
            "stimulus_direction",
            "stimulus_angular_position_degrees",
            "stimulus_angular_speed_degrees_per_second",
            "baseline_window_seconds",
        )
    )
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
                str(paper_spec["path"]): _sha256(paper_path),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "sources": config["source_types"],
        "array_axes": ["stimulus_on_off", "cell", "time_milliseconds"],
        "array_has_direction_axis": False,
        "paper_evidence": {
            "moving_edge_family": True,
            "ON_OFF_polarity": True,
            "speed_degrees_per_second": 30.0,
            "one_recorded_cell_per_different_animal": True,
            "responses_temporally_aligned_post_hoc": True,
        },
        "notebook_evidence": {
            "sample_rate_hz": 1000,
            "source_stimulus_axis_labels": ["on", "off"],
            "PD_ND_labels_apply_after_source_average_and_synthetic_shift": True,
        },
        "led_evidence": {
            "shape": list(led.shape),
            "dtype": str(led.dtype),
            "half_intensity_crossing_samples": crossings,
            "per_recording_stimulus_identifier_present": False,
            "per_recording_angular_position_present": False,
        },
        "field_status": field_status,
        "available_fields": available,
        "missing_fields": missing,
        "available_field_count": len(available),
        "required_field_count": len(required_fields),
        "all_required_recording_fields_available": not missing,
        "complete_stimulus_and_baseline_fields_on_allowed_payload": (
            complete_stimulus_and_baseline
        ),
        "authorize_source_dynamics_fit": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": "five_required_recording_fields_absent",
        "boundary": config["boundary"],
    }
