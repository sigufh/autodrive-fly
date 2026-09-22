"""Audit Henning C3 directional-edge data and its cohort boundaries."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import scipy.io
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-c3-directional-edge-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_c3_directional_edge_audit.py")


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _git_bytes(repository: Path, revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _verify_git_payload(repository: Path, revision: str, spec: dict) -> bytes:
    payload = _git_bytes(repository, revision, spec["path"])
    if (
        len(payload) != int(spec["bytes"])
        or _git_blob_sha1(payload) != spec["git_blob_sha1"]
        or hashlib.sha256(payload).hexdigest() != spec["sha256"]
    ):
        raise ValueError(f"C3 directional-edge payload identity changed: {spec['path']}")
    return payload


def _normalise_name(value: object) -> str:
    return str(value).split("_Image", 1)[0].removesuffix("_2")


def _xml_text(root: ElementTree.Element) -> str:
    return " ".join(" ".join(root.itertext()).split())


def evaluate_v7_c3_directional_edge_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository = root / config["repository"]["path"]
    revision = config["repository"]["archived_revision"]
    local_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_revision != revision:
        raise ValueError("C3 repository revision changed")

    source_payload = _verify_git_payload(repository, revision, config["source_data"])
    code_payload = _verify_git_payload(repository, revision, config["source_code"])
    code_text = code_payload.decode("utf-8")
    contract = config["author_analysis_contract"]
    required_code = (
        "theta=[90, 45, 0, 315, 270, 225, 180, 135, 90]",
        "OFF=max(C2C3(:,:,1:40),[],3)",
        "ON=max(C2C3(:,:,41:80),[],3)",
    )
    if any(fragment not in code_text for fragment in required_code):
        raise ValueError("C3 directional-edge analysis semantics changed")

    data = scipy.io.loadmat(
        io.BytesIO(source_payload),
        squeeze_me=True,
        struct_as_record=False,
    )
    records = np.atleast_1d(data[contract["source_struct"]])
    aggregate = data[contract["aggregate_struct"]]
    raw_shapes: dict[str, int] = {}
    for record in records:
        shape = "x".join(str(item) for item in np.asarray(record.iAV_ROI_resp).shape)
        raw_shapes[shape] = raw_shapes.get(shape, 0) + 1
    aggregate_values = np.asarray(
        getattr(aggregate, contract["aggregate_response_field"]), dtype=np.float64
    )
    names = sorted({_normalise_name(record.name) for record in records})
    fly_ids, roi_counts = np.unique(
        [int(record.flyID) for record in records], return_counts=True
    )
    expected = config["expected"]
    if (
        len(records) != int(expected["record_count"])
        or names != expected["normalized_fly_names"]
        or fly_ids.tolist() != expected["payload_fly_ids"]
        or list(aggregate_values.shape) != expected["aggregate_shape"]
        or not np.all(np.isfinite(aggregate_values))
    ):
        raise ValueError("C3 directional-edge inventory changed")

    xml_spec = config["paper"]["version_of_record_xml"]
    xml_path = root / xml_spec["path"]
    if (
        xml_path.stat().st_size != int(xml_spec["bytes"])
        or _sha256(xml_path) != xml_spec["sha256"]
    ):
        raise ValueError("C3 Version-of-Record XML identity changed")
    xml_root = ElementTree.fromstring(xml_path.read_bytes())
    doi = xml_root.findtext(".//article-id[@pub-id-type='doi']")
    text = _xml_text(xml_root)
    required_paper_phrases = (
        "GCaMP6f specifically in either one of the two cell types",
        "C3 (N=8 flies, 77 cells)",
        "showed no preference to any direction of motion",
        "velocity of 20°/s",
        "An eight-direction stimulus was shown to quantify direction-selectivity of C2 and C3",
        "interpolated at 10 Hz before averaging across flies",
    )
    if doi != config["paper"]["doi"] or any(
        phrase not in text for phrase in required_paper_phrases
    ):
        raise ValueError("C3 directional-edge paper evidence changed")

    related = {}
    for name, path_string in config["related_evidence"].items():
        path = root / path_string
        related[name] = json.loads(path.read_text(encoding="utf-8"))
    flash_names = set(
        related["flash"]["empirical_C3_flash"].get("fly_ids", [])
    )
    if not flash_names:
        flash_source = root / related["flash"]["protocol"]["source_data"]["path"]
        if _sha256(flash_source) != related["flash"]["protocol"]["source_data"][
            "sha256"
        ]:
            raise ValueError("C3 flash source differs from verified evidence")
        flash_records = np.atleast_1d(
            scipy.io.loadmat(flash_source, squeeze_me=True, struct_as_record=False)[
                related["flash"]["author_analysis_contract"]["source_struct"]
            ]
        )
        flash_names = {_normalise_name(record.name) for record in flash_records}
    strf_source = root / related["strf"]["protocol"]["source_data"]["path"]
    if _sha256(strf_source) != related["strf"]["protocol"]["source_data"][
        "sha256"
    ]:
        raise ValueError("C3 STRF source differs from verified evidence")
    strf_groups = scipy.io.loadmat(strf_source, simplify_cells=True)["RF_DATA"]
    strf_groups = strf_groups if isinstance(strf_groups, list) else list(strf_groups)
    c3_group = next(group for group in strf_groups if group["name"] == "C3")
    strf_names = {str(row["Flyname"]) for row in c3_group["DATA"]}
    directional_names = set(names)
    flash_overlap = sorted(directional_names & flash_names)
    strf_overlap = sorted(directional_names & strf_names)

    direction_degrees = [int(value) for value in contract["direction_degrees"]]
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                **{
                    path: _sha256(root / path)
                    for path in config["related_evidence"].values()
                },
            },
            "repository_url": config["repository"]["url"],
            "archived_revision": local_revision,
            "source_data": config["source_data"],
            "source_code": config["source_code"],
            "version_of_record_xml": xml_spec,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "measurement": {
            "source_type": "C3",
            "modality": "two_photon_GCaMP6f_calcium",
            "response_unit": contract["response_unit"],
            "record_count": len(records),
            "ROI_count": len(records),
            "fly_count_in_public_payload": len(names),
            "fly_count_in_version_of_record_caption": int(
                expected["paper_reported_fly_count"]
            ),
            "fly_ids": fly_ids.tolist(),
            "ROI_count_by_fly_id": {
                str(fly): int(count) for fly, count in zip(fly_ids, roi_counts, strict=True)
            },
            "normalized_fly_names": names,
            "raw_ROI_response_shapes": raw_shapes,
            "aggregate_shape": list(aggregate_values.shape),
            "aggregate_all_finite": bool(np.all(np.isfinite(aggregate_values))),
            "aggregate_value_range": [
                float(np.min(aggregate_values)),
                float(np.max(aggregate_values)),
            ],
            "direction_degrees": direction_degrees,
            "direction_count": len(direction_degrees),
            "edge_contrasts": ["OFF", "ON"],
            "edge_velocity_degrees_per_second": float(
                contract["edge_velocity_degrees_per_second"]
            ),
            "interpolated_sample_rate_hz": float(contract["interpolated_rate_hz"]),
            "interpolated_sample_interval_seconds": 1.0
            / float(contract["interpolated_rate_hz"]),
            "author_response_reduction": contract["response_reduction"],
            "paper_reports_direction_preference": False,
        },
        "cohort_relationship": {
            "flash_overlap_fly_names": flash_overlap,
            "flash_overlap_count": len(flash_overlap),
            "STRF_overlap_fly_names": strf_overlap,
            "STRF_overlap_count": len(strf_overlap),
            "every_directional_fly_in_flash_cohort": directional_names <= flash_names,
            "every_directional_fly_in_STRF_cohort": directional_names <= strf_names,
            "independent_study": False,
            "independent_cohort": False,
        },
        "source_contract_fields_added": {
            "stimulus_polarity": True,
            "stimulus_direction": True,
            "stimulus_angular_speed_degrees_per_second": True,
            "time_seconds": True,
            "sample_interval_seconds": True,
            "response_values": True,
            "biological_individual_id": True,
        },
        "experimental_membrane_voltage": False,
        "direction_specific_C3_source_kernel_verified": False,
        "recording_to_MaleCNS_body_crosswalk_found": False,
        "independent_dynamic_validation_available": False,
        "authorize_C3_source_dynamics_transfer": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "direction_fields_found_but_same_study_cohort_and_calcium_unit_"
            "do_not_satisfy_transfer_contract"
        ),
        "boundary": config["boundary"],
    }
