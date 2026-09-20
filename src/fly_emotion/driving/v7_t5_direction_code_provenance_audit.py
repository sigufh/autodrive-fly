"""Separate relative T5 labels from record-level physical direction provenance."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import (
    _load_restricted,
    _verify_file,
)
from fly_emotion.driving.v7_t5_conductance_audit import _load_cell, _vector

CONFIG = Path("configs/driving-v7-t5-direction-code-provenance-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_direction_code_provenance_audit.py")


def _scan_bundle(root: Path, spec: dict, tokens: list[str]) -> tuple[int, dict[str, list[str]]]:
    bundle = _verify_file(root, spec)
    with tempfile.TemporaryDirectory() as temporary:
        mirror = Path(temporary) / "motyxia2.git"
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--mirror", str(bundle), str(mirror)],
            check=False,
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            raise ValueError("Motyxia2 all-refs bundle could not be cloned")
        rev_list = subprocess.run(
            ["git", "-C", str(mirror), "rev-list", "--objects", "--all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths_by_object: dict[str, set[str]] = {}
        for row in rev_list:
            parts = row.split(" ", 1)
            if len(parts) == 2:
                paths_by_object.setdefault(parts[0], set()).add(parts[1])
        object_ids = sorted(paths_by_object)
        check = subprocess.run(
            [
                "git",
                "-C",
                str(mirror),
                "cat-file",
                "--batch-check=%(objectname) %(objecttype)",
            ],
            input="\n".join(object_ids) + "\n",
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        blob_ids = [row.split()[0] for row in check if row.endswith(" blob")]
        batch = subprocess.run(
            ["git", "-C", str(mirror), "cat-file", "--batch"],
            input=("\n".join(blob_ids) + "\n").encode(),
            check=True,
            capture_output=True,
        ).stdout
    stream = io.BytesIO(batch)
    hits = {token: set() for token in tokens}
    for expected_id in blob_ids:
        object_id, object_type, object_size = stream.readline().decode().strip().split()
        if object_id != expected_id or object_type != "blob":
            raise ValueError("Motyxia2 blob stream changed")
        payload = stream.read(int(object_size))
        if stream.read(1) != b"\n":
            raise ValueError("Motyxia2 blob delimiter changed")
        for token in tokens:
            if token.encode() in payload:
                hits[token].update(paths_by_object.get(object_id, set()))
    if stream.read():
        raise ValueError("Motyxia2 blob stream has trailing bytes")
    return len(blob_ids), {token: sorted(paths) for token, paths in hits.items()}


def _archive_texts(root: Path, stimulus_config: dict) -> dict[str, str]:
    archive_spec = stimulus_config["stimulus_repository"]["archive"]
    archive = _verify_file(root, archive_spec)
    texts = {}
    with tarfile.open(archive, "r:gz") as payload:
        for name, spec in stimulus_config["stimulus_repository"]["members"].items():
            matches = [
                member for member in payload.getmembers() if member.name.endswith(spec["suffix"])
            ]
            if len(matches) != 1:
                raise ValueError(f"Motyxia2 member changed: {name}")
            raw = payload.extractfile(matches[0]).read()
            if len(raw) != int(spec["bytes"]) or _sha256_bytes(raw) != spec["sha256"]:
                raise ValueError(f"Motyxia2 member payload changed: {name}")
            texts[name] = raw.decode("utf-8")
    return texts


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def evaluate_v7_t5_direction_code_provenance_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    label_config_path = Path(config["figure4_label_config"])
    label_evidence_path = Path(config["figure4_label_evidence"])
    identity_config_path = Path(config["kohn_portes_identity_config"])
    stimulus_config_path = Path(config["kohn_portes_stimulus_config"])
    history_evidence_path = Path(config["motyxia2_history_evidence"])
    history_config_path = Path(config["motyxia2_history_config"])
    label_config = yaml.safe_load((root / label_config_path).read_text())
    label = json.loads((root / label_evidence_path).read_text())
    figure4_repository_config_path = Path(config["figure4_repository_config"])
    figure4_repository_evidence_path = Path(config["figure4_repository_evidence"])
    figure4_repository_config = yaml.safe_load(
        (root / figure4_repository_config_path).read_text()
    )
    figure4_repository = json.loads(
        (root / figure4_repository_evidence_path).read_text()
    )
    identity_config = yaml.safe_load((root / identity_config_path).read_text())
    stimulus_config = yaml.safe_load((root / stimulus_config_path).read_text())
    history = json.loads((root / history_evidence_path).read_text())
    expected = config["expected"]

    relative_mapping = {
        int(key): value for key, value in label["label_status"]["direction_code_to_PD_ND"].items()
    }
    if relative_mapping != expected["figure4_relative_mapping"]:
        raise ValueError("Figure 4 relative direction mapping changed")
    repository_label = label["processed_repository_evidence"]
    if repository_label["direction_codes_present"] != expected["figure4_direction_codes"]:
        raise ValueError("Figure 4 direction-code values changed")
    if repository_label["cells_with_both_direction_codes"] != int(
        expected["figure4_cells_with_both_codes"]
    ):
        raise ValueError("Figure 4 direction-code cell count changed")

    extracted = {
        Path(item["path"]).name: item for item in label_config["figure4_dataset"]["extracted_files"]
    }
    organizing_path = root / extracted["organizingClusterData.m"]["path"]
    plotting_path = root / extracted["sourceDataPlottingFig4Script.m"]["path"]
    organizing = organizing_path.read_text(encoding="utf-8")
    plotting = plotting_path.read_text(encoding="utf-8")
    if "p.direction_mb'" not in organizing:
        raise ValueError("Figure 4 direction_mb extraction changed")
    relative_phrases = (
        "assert(relDirs(1) == 0 & relDirs(2) == 1, 'directions are flipped')",
        "dataND = tempDat.MB(relInds(1)).data;",
        "dataPD = tempDat.MB(relInds(2)).data;",
    )
    if any(phrase not in plotting for phrase in relative_phrases):
        raise ValueError("Figure 4 relative PD/ND mapping changed")
    absolute_direction_phrases = (
        "leftward",
        "rightward",
        "clockwise",
        "counterclockwise",
        "front-to-back",
        "back-to-front",
    )
    figure4_absolute_hits = [
        phrase
        for phrase in absolute_direction_phrases
        if phrase in (organizing + "\n" + plotting).lower()
    ]
    if figure4_absolute_hits:
        raise ValueError("Figure 4 code gained an unreviewed absolute direction label")

    repository_path = root / "data/raw/t5-conductance-model"
    repository_commit = subprocess.run(
        ["git", "-C", str(repository_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = figure4_repository_config["repository"]["commit"]
    if repository_commit != expected_commit:
        raise ValueError("Figure 4 local repository commit changed")
    if figure4_repository["repository"]["commit"] != expected_commit:
        raise ValueError("Figure 4 repository evidence commit changed")
    repository_files = {
        item["path"]: item for item in figure4_repository["repository"]["files"]
    }
    optimize_path = repository_path / "optimize_model.m"
    if _sha256(optimize_path) != repository_files["optimize_model.m"]["sha256"]:
        raise ValueError("Figure 4 moving-bar model code changed")
    optimize = optimize_path.read_text(encoding="utf-8")
    coordinate_fragments = (
        "p.pos = q.pos_mb;",
        "p.flag_dir = q.direction_mb(k3);",
        "if(p.flag_dir==1)",
        "eff_pos = [eff_pos p.pos(end)+i]",
        "if(p.flag_dir==0)",
        "eff_pos = fliplr(eff_pos);",
    )
    if any(fragment not in optimize for fragment in coordinate_fragments):
        raise ValueError("Figure 4 moving-bar coordinate semantics changed")
    cell_direction_counts = {}
    moving_bar_pair_count = 0
    for cell_id in range(1, int(figure4_repository_config["paper"]["recorded_cells"]) + 1):
        filename = f"data_cell_{cell_id}_all.mat"
        if _sha256(repository_path / filename) != repository_files[filename]["sha256"]:
            raise ValueError(f"Figure 4 moving-bar record changed: {filename}")
        _data, protocol = _load_cell(repository_path, cell_id, "all")
        directions = _vector(protocol.direction_mb).astype(int)
        positions = _vector(protocol.pos_mb).astype(float)
        counts = Counter(map(int, directions))
        if set(counts) != {0, 1} or not np.all(np.diff(positions) > 0):
            raise ValueError(f"Figure 4 direction/position semantics changed: {filename}")
        if counts[0] != counts[1]:
            raise ValueError(f"Figure 4 moving-bar directions are unpaired: {filename}")
        moving_bar_pair_count += counts[0]
        cell_direction_counts[str(cell_id)] = {str(code): counts[code] for code in (0, 1)}
    if moving_bar_pair_count != int(expected["figure4_moving_bar_pair_count"]):
        raise ValueError("Figure 4 moving-bar pair count changed")

    records = []
    historical_rows = [
        line.split("\t", 3)
        for line in _verify_file(root, identity_config["repository"]["historical_paths"])
        .read_text()
        .splitlines()
    ]
    pickle_history = {}
    for name, spec in identity_config["drifting_grating_files"].items():
        payload = _load_restricted(_verify_file(root, spec))
        if not isinstance(payload, list):
            raise ValueError(f"Kohn-Portes drifting-grating payload changed: {name}")
        records.extend(payload)
        historical_pickle_path = f"data/KohnPortes2021/{Path(spec['path']).name}"
        rows = [row for row in historical_rows if row[3] == historical_pickle_path]
        blobs = sorted({row[2] for row in rows})
        if blobs != [spec["git_blob"]] or len(rows) != int(
            expected["historical_commit_presence_per_pickle"]
        ):
            raise ValueError(f"Kohn-Portes drifting-grating history changed: {name}")
        pickle_history[name] = {
            "repository_path": historical_pickle_path,
            "commit_presence_count": len(rows),
            "unique_blob_count": len(blobs),
            "only_git_blob": blobs[0],
        }
    if len(records) != int(expected["kohn_portes_drifting_grating_record_count"]):
        raise ValueError("Kohn-Portes drifting-grating record count changed")
    if len(pickle_history) != int(expected["kohn_portes_drifting_grating_file_count"]):
        raise ValueError("Kohn-Portes drifting-grating file count changed")

    fields = sorted(set().union(*(set(record) for record in records)))
    direction_like = sorted(field for field in fields if "dire" in field.lower())
    nested_protocol_fields = sorted(
        {
            field
            for record in records
            for field, value in record.items()
            if isinstance(value, (dict, list, tuple))
            or (isinstance(value, np.ndarray) and value.dtype == object)
        }
    )
    if (
        direction_like
        or nested_protocol_fields
        or any("direction_mb" in record for record in records)
    ):
        raise ValueError("Kohn-Portes records gained direction or nested protocol metadata")
    field_values = {
        "direction_mb": {"present_count": 0, "missing_count": len(records)},
        "stimulus_path": {
            "present_count": sum("stimulus_path" in record for record in records),
            "unique_count": len({str(record["stimulus_path"]) for record in records}),
        },
        "stimulus_name": dict(Counter(str(record.get("stimulus_name")) for record in records)),
        "wn_orientation": dict(
            Counter(
                "null" if record.get("wn_orientation") is None else str(record["wn_orientation"])
                for record in records
            )
        ),
        "spatialfrequency_array": sorted(
            {tuple(np.asarray(record["spatialfrequency_array"], dtype=float)) for record in records}
        )[0],
        "temporalfrequency_array": sorted(
            {
                tuple(np.asarray(record["temporalfrequency_array"], dtype=float))
                for record in records
            }
        )[0],
    }

    texts = _archive_texts(root, stimulus_config)
    stimuli = texts["stimuli"]
    forms = texts["forms"]
    generator_phrases = (
        "dire_list=(0.,)",
        "'direction (deg)'",
        "for dire in self.dire_list",
        "dire=frame[4]",
        "distance * 2 * np.pi * spatial_freq - phase",
    )
    if any(phrase not in stimuli for phrase in generator_phrases):
        raise ValueError("Motyxia2 direction generation code changed")
    if "dire_list = ListField(" not in forms or "'directions (degrees)'" not in forms:
        raise ValueError("Motyxia2 direction form changed")

    bundle_spec = yaml.safe_load((root / history_config_path).read_text())["bundle"]
    blob_count, token_hits = _scan_bundle(root, bundle_spec, expected["motyxia2_history_tokens"])
    if blob_count != history["repository"]["unique_blob_count"]:
        raise ValueError("Motyxia2 history blob count changed")
    if any(token_hits.values()):
        raise ValueError("Motyxia2 history gained a direction-code provenance hit")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(label_config_path): _sha256(root / label_config_path),
                str(label_evidence_path): _sha256(root / label_evidence_path),
                str(figure4_repository_config_path): _sha256(
                    root / figure4_repository_config_path
                ),
                str(figure4_repository_evidence_path): _sha256(
                    root / figure4_repository_evidence_path
                ),
                "src/fly_emotion/driving/v7_t5_conductance_audit.py": _sha256(
                    root / "src/fly_emotion/driving/v7_t5_conductance_audit.py"
                ),
                str(identity_config_path): _sha256(root / identity_config_path),
                str(stimulus_config_path): _sha256(root / stimulus_config_path),
                str(history_evidence_path): _sha256(root / history_evidence_path),
                str(history_config_path): _sha256(root / history_config_path),
                identity_config["repository"]["historical_paths"]["path"]: _sha256(
                    root / identity_config["repository"]["historical_paths"]["path"]
                ),
                stimulus_config["stimulus_repository"]["archive"]["path"]: _sha256(
                    root / stimulus_config["stimulus_repository"]["archive"]["path"]
                ),
                bundle_spec["path"]: _sha256(root / bundle_spec["path"]),
                extracted["organizingClusterData.m"]["path"]: _sha256(organizing_path),
                extracted["sourceDataPlottingFig4Script.m"]["path"]: _sha256(plotting_path),
                "data/raw/t5-conductance-model/optimize_model.m": _sha256(
                    optimize_path
                ),
                **{
                    f"data/raw/t5-conductance-model/data_cell_{cell_id}_all.mat": (
                        _sha256(
                            repository_path / f"data_cell_{cell_id}_all.mat"
                        )
                    )
                    for cell_id in range(
                        1,
                        int(figure4_repository_config["paper"]["recorded_cells"])
                        + 1,
                    )
                },
                **{
                    spec["path"]: _sha256(root / spec["path"])
                    for spec in identity_config["drifting_grating_files"].values()
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "figure4_moving_bar_direction_code": {
            "dataset_scope": "Gruntman_et_al_Figure_4_T5_moving_bar",
            "direction_field": "direction_mb",
            "direction_codes_present": repository_label["direction_codes_present"],
            "cells_with_both_codes": repository_label["cells_with_both_direction_codes"],
            "relative_PD_ND_mapping_verified": True,
            "relative_mapping": {str(key): value for key, value in relative_mapping.items()},
            "record_level_native_coordinate_motion_mapping_verified": True,
            "native_coordinate_mapping": {
                "0": "decreasing_receptive_field_position_coordinate",
                "1": "increasing_receptive_field_position_coordinate",
            },
            "moving_bar_pair_count": moving_bar_pair_count,
            "per_record_direction_code_counts": cell_direction_counts,
            "absolute_screen_or_body_motion_mapping_verified": False,
            "absolute_direction_phrase_hits_in_extracted_analysis_code": figure4_absolute_hits,
            "interpretation": (
                "codes identify each cell's relative null/preferred response, "
                "not a shared physical screen direction"
            ),
        },
        "kohn_portes_drifting_grating_records": {
            "dataset_scope": "Kohn_Portes_Tm1_Tm2_Tm4_Tm9_whole_cell_drifting_grating",
            "record_count": len(records),
            "file_count": len(pickle_history),
            "all_fields": fields,
            "direction_like_fields": direction_like,
            "nested_protocol_metadata_fields": nested_protocol_fields,
            "selected_field_values": field_values,
            "pickle_history": pickle_history,
            "all_historical_pickle_paths_have_one_blob_version": True,
            "record_level_direction_code_present": False,
            "record_level_physical_direction_present": False,
        },
        "motyxia2_generator_and_history": {
            "commit": stimulus_config["stimulus_repository"]["commit"],
            "generator_direction_parameter_unit": "degrees",
            "generator_default_direction_degrees": 0.0,
            "direction_is_passed_to_grating_generator": True,
            "phase_increases_with_time": True,
            "audited_unique_blob_count": blob_count,
            "exact_token_path_hits": token_hits,
            "record_linked_direction_protocol_found": False,
            "generator_default_assigned_to_records": False,
        },
        "provenance_gates": {
            "figure4_relative_direction_code_to_PD_ND_verified": True,
            "figure4_direction_code_to_native_coordinate_motion_verified": True,
            "figure4_direction_code_to_absolute_physical_motion_verified": False,
            "kohn_portes_record_level_direction_code_available": False,
            "kohn_portes_record_level_physical_direction_available": False,
            "cross_dataset_direction_mapping_authorized": False,
            "generator_default_imputation_authorized": False,
        },
        "direction_code_provenance_complete_for_kohn_portes": False,
        "authorize_kohn_portes_direction_conditioning": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "relative_Figure4_PD_ND_codes_do_not_recover_"
            "Kohn_Portes_record_level_physical_direction"
        ),
        "boundary": config["boundary"],
    }
