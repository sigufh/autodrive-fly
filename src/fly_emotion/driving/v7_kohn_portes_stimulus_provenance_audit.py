"""Audit record-level stimulus provenance for Kohn-Portes T5 voltages."""

from __future__ import annotations

import hashlib
import tarfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)

CONFIG = Path("configs/driving-v7-kohn-portes-stimulus-provenance-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_kohn_portes_stimulus_provenance_audit.py"
)


def _verify_archive(root: Path, repository: dict) -> tuple[dict[str, str], int]:
    spec = repository["archive"]
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
        raise ValueError("Motyxia2 archive changed")
    texts = {}
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {member.name.split("/", 1)[0] for member in members}
        expected_root = f"motyxia2-{repository['commit']}"
        if roots != {expected_root}:
            raise ValueError("Motyxia2 archive root does not bind the declared commit")
        files = [member for member in members if member.isfile()]
        if len(files) != int(spec["file_count"]):
            raise ValueError("Motyxia2 archive file count changed")
        for name, member_spec in repository["members"].items():
            matches = [member for member in files if member.name.endswith(member_spec["suffix"])]
            if len(matches) != 1:
                raise ValueError(f"Motyxia2 archive member changed: {name}")
            payload = archive.extractfile(matches[0]).read()
            if len(payload) != int(member_spec["bytes"]):
                raise ValueError(f"Motyxia2 member size changed: {name}")
            if hashlib.sha256(payload).hexdigest() != member_spec["sha256"]:
                raise ValueError(f"Motyxia2 member SHA-256 changed: {name}")
            texts[name] = payload.decode("utf-8")
    return texts, len(files)


def _load_records(root: Path, specs: dict) -> list[dict]:
    records = []
    for spec in specs.values():
        path = _verify_file(root, spec)
        payload = _load_restricted(path)
        if not isinstance(payload, list):
            raise ValueError(f"Kohn-Portes raw record payload changed: {path}")
        records.extend(payload)
    return records


def _field_inventory(records: list[dict]) -> dict:
    intersection = set(records[0])
    union = set()
    for record in records:
        intersection &= set(record)
        union |= set(record)
    return {
        "common_fields": sorted(intersection),
        "all_fields": sorted(union),
        "direction_like_fields": sorted(
            key for key in union if "dire" in key.lower()
        ),
        "position_or_center_fields": sorted(
            key for key in union if "position" in key.lower() or "center" in key.lower()
        ),
        "baseline_fields": sorted(key for key in union if "baseline" in key.lower()),
    }


def evaluate_v7_kohn_portes_stimulus_provenance_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    paper = config["paper"]
    paper_path = root / paper["path"]
    if paper_path.stat().st_size != int(paper["bytes"]) or _sha256(paper_path) != paper["sha256"]:
        raise ValueError("Kohn-Portes paper source changed")
    paper_root = ElementTree.parse(paper_path).getroot()
    paper_text = " ".join(
        (element.text or "") for element in paper_root.findall(".//text")
    )
    paper_phrases = (
        "The stimulus was randomly generated for each presentation",
        "5° horizontal bars flickering at 60 Hz",
        "0.5 Hz, high contrast drifting square waves",
        (
            "spatial wavelengths ranging from 2.5°, 10°, 12.5°, 25°, "
            "40°, 50°, 80°, 100°, 125°, and 200°"
        ),
        "https://gitlab.com/rbehnialab/motyxia2/-/tree/whitenoise",
    )
    if any(phrase not in paper_text for phrase in paper_phrases):
        raise ValueError("Kohn-Portes stimulus Methods text changed")

    repository = config["stimulus_repository"]
    texts, archive_file_count = _verify_archive(root, repository)
    stimuli = texts["stimuli"]
    forms = texts["forms"]
    generator_phrases = (
        "class WhiteNoise(LocallySparseNoise):",
        "np.random.shuffle(frames)",
        "class DriftingGratingCircle(Stim):",
        "'direction (deg)'",
        "'spatial frequency (cycle/deg)'",
        "'temporal frequency (Hz)'",
    )
    if any(phrase not in stimuli for phrase in generator_phrases):
        raise ValueError("Motyxia2 stimulus-generator semantics changed")
    form_phrases = ("class WhiteNoiseForm", "class DriftingGratingCircleForm")
    if any(phrase not in forms for phrase in form_phrases):
        raise ValueError("Motyxia2 stimulus form definitions changed")
    if "np.random.seed" in stimuli or "random.seed" in stimuli:
        raise ValueError("Motyxia2 revision now exposes an unreviewed random seed")

    source_configs = {
        name: yaml.safe_load((root / path).read_text(encoding="utf-8"))
        for name, path in config["source_configs"].items()
    }
    white_noise = _load_records(root, source_configs["ephys"]["white_noise_files"])
    drifting = _load_records(root, source_configs["identity"]["drifting_grating_files"])
    expected = config["expected"]
    required_sources = set(expected["source_types"])
    for name, records in (("white_noise", white_noise), ("drifting_grating", drifting)):
        if {str(record["cell_type"]) for record in records} != required_sources:
            raise ValueError(f"Kohn-Portes {name} source coverage changed")
        if any(
            not str(record["stimulus_path"]).startswith(expected["private_stimulus_path_prefix"])
            for record in records
        ):
            raise ValueError(f"Kohn-Portes {name} stimulus path provenance changed")

    wn_names = sorted({str(record["stimulus_name"]) for record in white_noise})
    wn_orientations = Counter(
        "null" if record.get("wn_orientation") is None else str(record["wn_orientation"])
        for record in white_noise
    )
    dg_names = sorted({str(record["stimulus_name"]) for record in drifting})
    sf_arrays = {
        tuple(np.asarray(record["spatialfrequency_array"], dtype=float))
        for record in drifting
    }
    tf_arrays = {
        tuple(np.asarray(record["temporalfrequency_array"], dtype=float))
        for record in drifting
    }
    checks = {
        "white_noise_record_count": len(white_noise),
        "white_noise_unique_stimulus_path_count": len(
            {str(record["stimulus_path"]) for record in white_noise}
        ),
        "white_noise_stimulus_names": wn_names,
        "white_noise_orientation_counts": dict(sorted(wn_orientations.items())),
        "drifting_grating_record_count": len(drifting),
        "drifting_grating_unique_stimulus_path_count": len(
            {str(record["stimulus_path"]) for record in drifting}
        ),
        "drifting_grating_stimulus_names": dg_names,
    }
    for key, actual in checks.items():
        if actual != expected[key]:
            raise ValueError(f"Kohn-Portes stimulus record summary changed: {key}")
    if sf_arrays != {tuple(expected["drifting_grating_spatial_frequencies_cycles_per_degree"])}:
        raise ValueError("Kohn-Portes drifting-grating spatial frequencies changed")
    if tf_arrays != {tuple(expected["drifting_grating_temporal_frequencies_hz"])}:
        raise ValueError("Kohn-Portes drifting-grating temporal frequencies changed")

    wn_inventory = _field_inventory(white_noise)
    dg_inventory = _field_inventory(drifting)
    if dg_inventory["direction_like_fields"]:
        raise ValueError("Kohn-Portes drifting-grating records gained a direction field")
    record_logs_local = all(
        (root / str(record["stimulus_path"])).is_file()
        for record in white_noise + drifting
    )
    if record_logs_local:
        raise ValueError("private record-specific stimulus logs unexpectedly became local")

    field_status = {
        "raw_white_noise": {
            "stimulus_direction": {
                "available": False,
                "reason": "random_flicker_has_no_motion_direction",
            },
            "stimulus_angular_position_degrees": {
                "available": False,
                "reason": "random_frame_log_is_only_an_unavailable_private_path",
            },
            "stimulus_angular_speed_degrees_per_second": {
                "available": False,
                "reason": "not_a_motion_stimulus",
            },
            "baseline_window_seconds": {
                "available": False,
                "reason": "not_retained_and_generator_defaults_are_not_record_provenance",
            },
        },
        "raw_drifting_grating": {
            "stimulus_direction": {
                "available": False,
                "reason": (
                    "generator_supports_direction_but_record_has_no_"
                    "direction_field_or_public_log"
                ),
            },
            "stimulus_angular_position_degrees": {
                "available": False,
                "reason": (
                    "generator_supports_center_but_record_has_no_"
                    "center_field_or_public_log"
                ),
            },
            "stimulus_angular_speed_degrees_per_second": {
                "available": True,
                "reason": "record_retains_temporal_and_spatial_frequency_arrays",
            },
            "baseline_window_seconds": {
                "available": False,
                "reason": "not_retained_and_generator_defaults_are_not_record_provenance",
            },
        },
    }
    gates = {
        "paper_link_to_whitenoise_branch_verified": True,
        "fixed_stimulus_generator_revision_verified": True,
        "record_specific_stimulus_logs_locally_available": record_logs_local,
        "white_noise_random_frame_sequence_reconstructible": False,
        "drifting_grating_direction_recoverable_per_recording": False,
        "drifting_grating_center_recoverable_per_recording": False,
        "baseline_window_recoverable_per_recording": False,
    }
    complete = all(gates.values())
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                paper["path"]: _sha256(paper_path),
                repository["archive"]["path"]: _sha256(
                    root / repository["archive"]["path"]
                ),
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
                **{path: _sha256(root / path) for path in config["source_configs"].values()},
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "stimulus_repository": {
            "url": repository["url"],
            "branch": repository["branch"],
            "commit": repository["commit"],
            "archive": {**repository["archive"], "actual_file_count": archive_file_count},
            "member_sha256": {
                name: spec["sha256"] for name, spec in repository["members"].items()
            },
            "white_noise_uses_unseeded_random_generation_and_shuffle": True,
            "drifting_grating_generator_exposes_direction_center_radius_and_frequency": True,
        },
        "raw_record_summary": {
            **checks,
            "private_stimulus_path_prefix": expected["private_stimulus_path_prefix"],
            "record_specific_stimulus_logs_locally_available": record_logs_local,
            "white_noise_field_inventory": wn_inventory,
            "drifting_grating_field_inventory": dg_inventory,
            "drifting_grating_spatial_frequencies_cycles_per_degree": list(next(iter(sf_arrays))),
            "drifting_grating_temporal_frequencies_hz": list(next(iter(tf_arrays))),
        },
        "recording_field_status": field_status,
        "transfer_gates": gates,
        "stimulus_provenance_contract_complete": complete,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": "generator_code_does_not_recover_missing_record_specific_stimulus_logs",
        "boundary": config["boundary"],
    }
