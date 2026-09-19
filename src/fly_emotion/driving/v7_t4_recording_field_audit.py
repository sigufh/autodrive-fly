"""Audit Fig. 3 T4 source payload fields against the frozen external contract."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

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
        for name in (
            "contract",
            "retrieval_evidence",
            "identity_evidence",
            "split_evidence",
        )
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
        "after subtracting a 1 s prestimulus baseline",
    )
    if not all(phrase in paper_text for phrase in required_paper_phrases):
        raise ValueError("T4 paper stimulus or identity statement changed")

    workbook_spec = config["source_workbook"]
    workbook_path = root / workbook_spec["path"]
    if (
        workbook_path.stat().st_size != int(workbook_spec["bytes"])
        or _sha256(workbook_path) != workbook_spec["sha256"]
    ):
        raise ValueError("T4 Fig. 3 source workbook changed")
    with zipfile.ZipFile(workbook_path) as workbook_archive:
        member_names = workbook_archive.namelist()
        connection_root = ElementTree.fromstring(
            workbook_archive.read("xl/connections.xml")
        )
        workbook_root = ElementTree.fromstring(workbook_archive.read("xl/workbook.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    connections = connection_root.findall("x:connection", namespace)
    connection_names = [item.attrib["name"] for item in connections]
    raw_connections = sorted(
        name for name in connection_names if name.startswith("sd_fig3a_")
    )
    derived_connections = sorted(
        name for name in connection_names if name.startswith("sd_fig3b_")
    )
    if len(connections) != int(workbook_spec["expected_connection_count"]):
        raise ValueError("T4 Fig. 3 workbook connection count changed")
    if raw_connections != sorted(workbook_spec["expected_raw_connection_names"]):
        raise ValueError("T4 Fig. 3 raw connection names changed")
    if len(derived_connections) != len(connections) - len(raw_connections):
        raise ValueError("T4 Fig. 3 derived connection names changed")
    if any(item.attrib.get("deleted") != "1" for item in connections):
        raise ValueError("T4 Fig. 3 workbook gained a live external connection")
    query_tables = [
        name
        for name in member_names
        if name.startswith("xl/queryTables/queryTable") and name.endswith(".xml")
    ]
    if len(query_tables) != int(workbook_spec["expected_query_table_count"]):
        raise ValueError("T4 Fig. 3 workbook query-table count changed")
    sheet_elements = workbook_root.findall("x:sheets/x:sheet", namespace)
    sheet_names = [sheet.attrib["name"] for sheet in sheet_elements]
    if sheet_names != workbook_spec["expected_sheets"]:
        raise ValueError("T4 Fig. 3 workbook sheet names changed")
    hidden_sheets = [
        sheet.attrib["name"]
        for sheet in sheet_elements
        if sheet.attrib.get("state", "visible") != "visible"
    ]
    package_metadata_members = sorted(
        name
        for name in member_names
        if "comment" in name.lower()
        or name == "docProps/custom.xml"
        or name.startswith("xl/externalLinks/")
    )

    manifest_spec = config["dataset_manifest"]
    manifest_path = root / manifest_spec["path"]
    if (
        manifest_path.stat().st_size != int(manifest_spec["bytes"])
        or _sha256(manifest_path) != manifest_spec["sha256"]
    ):
        raise ValueError("T4 Edmond dataset manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_files = manifest["data"]["latestVersion"]["files"]
    if len(dataset_files) != int(manifest_spec["expected_dataset_file_count"]):
        raise ValueError("T4 Edmond dataset file count changed")
    fig3_files = sorted(
        item["dataFile"]["filename"]
        for item in dataset_files
        if item.get("directoryLabel", "").endswith("/Fig. 3")
    )
    if fig3_files != sorted(manifest_spec["expected_fig3_files"]):
        raise ValueError("T4 Edmond Fig. 3 directory membership changed")
    identity_pattern = re.compile(
        r"animal|fly|individual|metadata|recording|specimen|subject", re.I
    )
    fig3_identity_sidecars = [
        name for name in fig3_files if identity_pattern.search(name)
    ]

    metadata_spec = config["dataset_notebook_metadata"]
    metadata_directory = root / metadata_spec["directory"]
    metadata_manifest = {
        item["dataFile"]["filename"]: item["dataFile"]
        for item in dataset_files
        if item["dataFile"]["filename"].endswith(
            (".ipynb", ".txt", ".yml")
        )
    }
    if len(metadata_manifest) != int(metadata_spec["expected_file_count"]):
        raise ValueError("T4 Edmond notebook/metadata manifest count changed")
    metadata_files = {}
    notebook_sources = {}
    for name, data_file in sorted(metadata_manifest.items()):
        path = metadata_directory / name
        if (
            path.stat().st_size != int(data_file["filesize"])
            or _md5(path) != data_file["md5"]
        ):
            raise ValueError(f"T4 Edmond notebook/metadata file changed: {name}")
        metadata_files[name] = {
            "datafile_id": int(data_file["id"]),
            "bytes": path.stat().st_size,
            "md5": data_file["md5"],
            "sha256": _sha256(path),
        }
        if name.endswith(".ipynb"):
            notebook_sources[name] = _notebook_source(path)
    if sum(item["bytes"] for item in metadata_files.values()) != int(
        metadata_spec["expected_total_bytes"]
    ):
        raise ValueError("T4 Edmond notebook/metadata byte total changed")
    fig3_source_array_names = set(metadata_spec["fig3_source_array_names"])
    consumer_notebooks = {}
    referenced_file_pattern = re.compile(
        r"['\"]([^'\"]+\.(?:npy|csv|xlsx|txt|pkl|pickle|h5|hdf5|tif))['\"]",
        re.I,
    )
    for name, source in notebook_sources.items():
        references = sorted(set(referenced_file_pattern.findall(source)))
        source_references = sorted(
            array_name for array_name in fig3_source_array_names if array_name in source
        )
        if not source_references:
            continue
        direct_loads = sorted(
            set(
                re.findall(
                    r"np\.load\(['\"](fig3_(?:C3|Mi1|Mi4|Mi9|Tm3)\.npy)['\"]\)",
                    source,
                )
            )
        )
        identity_sidecar_references = sorted(
            reference for reference in references if identity_pattern.search(reference)
        )
        consumer_notebooks[name] = {
            "fig3_source_array_references": source_references,
            "direct_numpy_load_references": direct_loads,
            "identity_or_recording_sidecar_references": identity_sidecar_references,
            "averages_source_arrays_over_cell_axis": "np.nanmean" in source
            and "axis=1" in source.replace(" ", ""),
        }
    if sorted(consumer_notebooks) != sorted(
        metadata_spec["expected_consumer_notebooks"]
    ):
        raise ValueError("T4 Fig. 3 notebook consumer set changed")
    if any(
        item["direct_numpy_load_references"]
        != item["fig3_source_array_references"]
        or item["identity_or_recording_sidecar_references"]
        or not item["averages_source_arrays_over_cell_axis"]
        for item in consumer_notebooks.values()
    ):
        raise ValueError("T4 Fig. 3 notebook metadata semantics changed")

    fig1_path = Path(config["fig1_identity_evidence"])
    fig1 = json.loads((root / fig1_path).read_text(encoding="utf-8"))
    fig1_counts = fig1["extended_figure_1"]["individual_source_spatial_RF_counts"]
    fig3_counts = {
        source: int(identity["source_summary"][source]["cell_count"])
        for source in config["source_types"]
    }

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
                str(workbook_spec["path"]): _sha256(workbook_path),
                str(manifest_spec["path"]): _sha256(manifest_path),
                str(fig1_path): _sha256(root / fig1_path),
                **{
                    str(metadata_directory / name): item["sha256"]
                    for name, item in metadata_files.items()
                },
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
            "one_second_prestimulus_baseline_applies_to_PD_grating_protocol": True,
            "Fig3_edge_recording_baseline_window_declared": False,
        },
        "complete_public_dataset_index": {
            "dataset_file_count": len(dataset_files),
            "Fig3_directory_files": fig3_files,
            "Fig3_directory_file_count": len(fig3_files),
            "Fig3_identity_or_metadata_sidecars": fig3_identity_sidecars,
        },
        "source_workbook_package": {
            "sheet_names": sheet_names,
            "hidden_sheets": hidden_sheets,
            "connection_count": len(connections),
            "all_connections_deleted": all(
                item.attrib.get("deleted") == "1" for item in connections
            ),
            "raw_input_connection_names": raw_connections,
            "derived_connection_count": len(derived_connections),
            "package_metadata_members": package_metadata_members,
            "recording_metadata_recovered_from_package": False,
        },
        "cross_directory_notebook_audit": {
            "verified_file_count": len(metadata_files),
            "verified_total_bytes": sum(
                item["bytes"] for item in metadata_files.values()
            ),
            "files": metadata_files,
            "Fig3_source_consumer_notebooks": consumer_notebooks,
            "Fig3_source_consumer_notebook_count": len(consumer_notebooks),
            "identity_or_recording_sidecar_reference_found": False,
            "recording_metadata_recovered_from_other_notebooks": False,
        },
        "cross_figure_identity_boundary": {
            "Fig1_individual_spatial_RF_counts": {
                source: int(fig1_counts[source]) for source in config["source_types"]
            },
            "Fig3_voltage_cell_counts": fig3_counts,
            "shared_stable_individual_identifier_present": False,
            "cohort_overlap_or_disjointness_identifiable": False,
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
