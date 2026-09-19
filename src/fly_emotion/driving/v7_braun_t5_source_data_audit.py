"""Audit Braun et al. Tm2, Tm9, and CT1 calcium source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-braun-t5-source-data-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_braun_t5_source_data_audit.py")


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value):
    return value.decode() if isinstance(value, bytes) else float(value)


def _inspect_hdf(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        if list(source) != ["data"]:
            raise ValueError("Braun HDF5 root changed")
        data = source["data"]
        level_count = int(data.attrs["axis1_nlevels"])
        names = []
        levels = []
        codes = []
        for index in range(level_count):
            level = data[f"axis1_level{index}"]
            name = next(
                value.decode()
                for key, value in level.attrs.items()
                if str(key).startswith("axis1_name")
            )
            names.append(name)
            levels.append([_decode(value) for value in level[:]])
            codes.append(data[f"axis1_label{index}"][:])
        if any(
            int(code.min()) < 0 or int(code.max()) >= len(levels[index])
            for index, code in enumerate(codes)
        ):
            raise ValueError("Braun HDF5 contains invalid or missing index codes")
        used = {
            name: [levels[index][value] for value in sorted(set(codes[index].tolist()))]
            for index, name in enumerate(names)
        }
        position = {name: index for index, name in enumerate(names)}
        fly_roi = np.unique(
            np.column_stack((codes[position["fly_id"]], codes[position["roi_index"]])),
            axis=0,
        )
        condition = np.unique(
            np.column_stack(
                [
                    codes[position[name]]
                    for name in (
                        "fly_id",
                        "roi_index",
                        "edge_intensity",
                        "rotation [deg]",
                        "trial",
                    )
                ]
            ),
            axis=0,
        )
        time = np.asarray(used["time"], dtype=np.float64)
        values = data["block0_values"][:, 0]
        return {
            "row_count": int(values.size),
            "index_names": names,
            "unique_values": {name: value for name, value in used.items() if name != "time"},
            "fly_count": len(used["fly_id"]),
            "fly_roi_pair_count": len(fly_roi),
            "condition_tuple_count": len(condition),
            "full_condition_grid": len(condition) == len(fly_roi) * 2 * 4 * 3,
            "time_count": len(time),
            "time_start_seconds": float(time.min()),
            "time_stop_seconds": float(time.max()),
            "median_time_step_seconds": float(np.median(np.diff(np.sort(time)))),
            "response_column": _decode(data["axis0"][0]),
            "finite_fraction": float(np.isfinite(values).mean()),
            "minimum_value": float(values.min()),
            "maximum_value": float(values.max()),
        }


def evaluate_v7_braun_t5_source_data_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    required_sources = set(contract["required_families"]["T5"]["source_types"])
    if not set(config["payloads"]).issubset(required_sources):
        raise ValueError("Braun payload includes a source outside the T5 contract")
    manifest_spec = config["dataset"]["manifest"]
    manifest_path = root / manifest_spec["path"]
    if (
        manifest_path.stat().st_size != int(manifest_spec["bytes"])
        or _sha256(manifest_path) != manifest_spec["sha256"]
    ):
        raise ValueError("Braun Edmond manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["data"]
    version = manifest["latestVersion"]
    if (
        int(manifest["id"]) != int(config["dataset"]["dataset_id"])
        or int(version["id"]) != int(config["dataset"]["version_id"])
        or int(version["versionNumber"]) != int(config["dataset"]["version_number"])
        or len(version["files"]) != int(config["dataset"]["file_count"])
        or sum(int(item["dataFile"]["filesize"]) for item in version["files"])
        != int(config["dataset"]["total_bytes"])
    ):
        raise ValueError("Braun Edmond dataset identity changed")
    manifest_by_id = {int(item["dataFile"]["id"]): item["dataFile"] for item in version["files"]}

    paper_spec = config["paper_pdf"]
    paper_path = root / paper_spec["path"]
    if (
        paper_path.stat().st_size != int(paper_spec["bytes"])
        or _sha256(paper_path) != paper_spec["sha256"]
    ):
        raise ValueError("Braun paper PDF changed")
    paper = PdfReader(paper_path)
    if len(paper.pages) != int(paper_spec["pages"]) or paper.attachments:
        raise ValueError("Braun paper PDF structure changed")
    paper_text = " ".join(" ".join((page.extract_text() or "").split()) for page in paper.pages)
    paper_text = paper_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    for phrase in (
        "Raw and Analyzed Data This paper https://doi.org/10.17617/3.QE3MFT",
        "Normalized two-photon calcium activity of Tm2",
        "Tm9",
        "CT1",
        "Each stimulus was repeated three to five times",
        "Regions of interest (ROIs) were drawn manually",
    ):
        if phrase not in paper_text:
            raise ValueError(f"Braun paper evidence changed: {phrase}")

    dependencies = {
        str(CONFIG): _sha256(root / CONFIG),
        str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
        str(contract_path): _sha256(root / contract_path),
        str(manifest_spec["path"]): _sha256(manifest_path),
        str(paper_spec["path"]): _sha256(paper_path),
    }
    payload_reports = {}
    for source_name, spec in config["payloads"].items():
        path = root / spec["path"]
        manifest_file = manifest_by_id[int(spec["file_id"])]
        if (
            manifest_file["filename"] != path.name
            or int(manifest_file["filesize"]) != int(spec["bytes"])
            or manifest_file["checksum"]["value"] != spec["publisher_md5"]
            or path.stat().st_size != int(spec["bytes"])
            or _md5(path) != spec["publisher_md5"]
            or _sha256(path) != spec["sha256"]
        ):
            raise ValueError(f"Braun payload identity changed: {source_name}")
        report = _inspect_hdf(path)
        if report["index_names"] != config["expected_index_names"]:
            raise ValueError(f"Braun index schema changed: {source_name}")
        for field, expected in config["expected_common_values"].items():
            if report["unique_values"][field] != expected:
                raise ValueError(f"Braun condition field changed: {source_name}:{field}")
        expected_values = {
            "row_count": spec["row_count"],
            "fly_count": spec["fly_count"],
            "fly_roi_pair_count": spec["fly_roi_pair_count"],
            "time_count": spec["time_count"],
        }
        if any(report[name] != int(value) for name, value in expected_values.items()):
            raise ValueError(f"Braun payload counts changed: {source_name}")
        if (
            not report["full_condition_grid"]
            or report["finite_fraction"] != 1.0
            or not np.isclose(
                report["median_time_step_seconds"],
                config["expected_time_step_seconds"],
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"Braun payload completeness changed: {source_name}")
        payload_reports[source_name] = {
            **spec,
            **report,
            "response_unit": "deltaF_over_F",
            "indicator": "GCaMP7f",
            "stable_pseudonymous_fly_IDs_present": True,
            "paper_and_payload_fly_counts_match": (
                report["fly_count"] == int(spec["paper_figure_5_fly_count"])
            ),
        }
        dependencies[spec["path"]] = _sha256(path)

    notebook_reports = {}
    for name, spec in config["notebooks"].items():
        path = root / spec["path"]
        manifest_file = manifest_by_id[int(spec["file_id"])]
        if (
            manifest_file["filename"] != path.name
            or manifest_file["checksum"]["value"] != spec["publisher_md5"]
            or path.stat().st_size != int(spec["bytes"])
            or _md5(path) != spec["publisher_md5"]
            or _sha256(path) != spec["sha256"]
        ):
            raise ValueError(f"Braun notebook identity changed: {name}")
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if cell.get("cell_type") == "code"
        )
        notebook_reports[name] = {
            **spec,
            "reads_HDF_payload": "pd.read_hdf" in source,
            "groups_by_fly_ID": "fly_id" in source,
            "declares_baseline_window": "baseline" in source.lower(),
        }
        dependencies[spec["path"]] = _sha256(path)

    source_fields = {
        name: {
            "stable_biological_individual_id": True,
            "stable_recording_unit_id": True,
            "stimulus_id": True,
            "stimulus_family": True,
            "stimulus_polarity": True,
            "stimulus_direction_rotation_degrees": True,
            "stimulus_angular_speed_degrees_per_second": True,
            "time_seconds": True,
            "response_values": True,
            "response_unit_allowed": False,
            "baseline_window_seconds": False,
            "source_to_MaleCNS_mapping": False,
            "required_split_roles": False,
        }
        for name in payload_reports
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": dependencies,
            "paper": config["paper"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "dataset": {
            **config["dataset"],
            "manifest_file_count_verified": True,
            "all_selected_files_match_publisher_MD5": True,
        },
        "payloads": payload_reports,
        "notebooks": notebook_reports,
        "source_field_status": source_fields,
        "sources_with_local_temporal_calcium": sorted(payload_reports),
        "sources_with_stable_pseudonymous_fly_IDs": sorted(payload_reports),
        "sources_with_complete_condition_grid": sorted(payload_reports),
        "sources_with_allowed_response_unit": [],
        "all_payload_fly_counts_match_paper_figure_5": all(
            item["paper_and_payload_fly_counts_match"] for item in payload_reports.values()
        ),
        "baseline_window_declared": False,
        "source_to_MaleCNS_mapping_available": False,
        "required_split_roles_available": False,
        "authorize_source_dynamics_fit": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "rich_Tm2_Tm9_CT1_calcium_payload_is_not_allowed_voltage_and_lacks_"
            "baseline_mapping_and_split_roles"
        ),
        "boundary": config["boundary"],
    }
