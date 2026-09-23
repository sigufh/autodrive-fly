"""Audit Drews Mi4 calcium data and bound Sporar C3 context mentions."""

from __future__ import annotations

import hashlib
import io
import math
import pickle
import pickletools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index, _new_Index
from pandas.core.indexes.multi import MultiIndex
from pandas.core.internals.managers import BlockManager
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-drews-mi4-sporar-c3-boundary-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_drews_mi4_sporar_c3_boundary_audit.py"
)


class _FrozenNDArray(np.ndarray):
    """Compatibility target for the immutable ndarray used by legacy pandas."""

    def __new__(cls, data, dtype=None, copy=False):
        return np.array(data, dtype=dtype, copy=copy).view(cls)


class _RestrictedPandasUnpickler(pickle.Unpickler):
    """Allow only the fixed pandas/NumPy containers present in the payload."""

    _allowed = {
        ("__builtin__", "slice"): slice,
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("pandas.core.frame", "DataFrame"): DataFrame,
        ("pandas.core.indexes.base", "Index"): Index,
        ("pandas.core.indexes.base", "_new_Index"): _new_Index,
        ("pandas.core.indexes.frozen", "FrozenNDArray"): _FrozenNDArray,
        ("pandas.core.indexes.multi", "MultiIndex"): MultiIndex,
        ("pandas.core.indexes.numeric", "Float64Index"): Index,
        ("pandas.core.indexes.numeric", "Int64Index"): Index,
        ("pandas.core.internals.managers", "BlockManager"): BlockManager,
    }

    def find_class(self, module: str, name: str):
        try:
            return self._allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"forbidden pickle global: {module}.{name}"
            ) from exc


def _git_blob(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload, usedforsecurity=False).hexdigest()


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"file size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"file SHA-256 mismatch: {path}")
    if "git_blob" in spec and _git_blob(path.read_bytes()) != spec["git_blob"]:
        raise ValueError(f"Git blob mismatch: {path}")
    return path


def _term_inventory(pages: list[str], term: str) -> tuple[list[int], int]:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])")
    counts = [len(pattern.findall(page)) for page in pages]
    return [index for index, count in enumerate(counts, 1) if count], sum(counts)


def _verify_term_inventory(pages: list[str], spec: dict, label: str) -> dict:
    inventory = {}
    for source in ("Mi4", "C3"):
        hit_pages, hit_count = _term_inventory(pages, source)
        if hit_pages != spec[f"{source}_exact_term_pages"] or hit_count != int(
            spec[f"{source}_exact_term_count"]
        ):
            raise ValueError(f"{label} {source} term inventory changed")
        inventory[source] = {
            "exact_term_pages": hit_pages,
            "exact_term_count": hit_count,
        }
    return inventory


def _load_restricted(path: Path) -> tuple[pd.DataFrame, list[str]]:
    payload = path.read_bytes()
    globals_used = sorted(
        str(value)
        for opcode, value, _ in pickletools.genops(payload)
        if opcode.name == "GLOBAL"
    )
    allowed_globals = {
        f"{module} {name}" for module, name in _RestrictedPandasUnpickler._allowed
    }
    forbidden = set(globals_used) - allowed_globals
    if forbidden:
        raise ValueError(f"Drews pickle has forbidden globals: {sorted(forbidden)}")
    frame = _RestrictedPandasUnpickler(io.BytesIO(payload), encoding="latin1").load()
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Drews Figure 3 payload is not a DataFrame")
    return frame, globals_used


def evaluate_v7_drews_mi4_sporar_c3_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    drews = config["drews_2020"]
    dissertation = drews["dissertation"]
    landing = _verify_file(root, dissertation["landing_page"])
    dissertation_path = _verify_file(root, dissertation["pdf"])
    if hashlib.md5(dissertation_path.read_bytes()).hexdigest() != dissertation["pdf"][
        "md5"
    ]:
        raise ValueError("Schützenberger repository MD5 changed")
    landing_text = landing.read_text(encoding="utf-8")
    if (
        dissertation["title"] not in landing_text
        or "Schützenberger, Anna" not in landing_text
        or dissertation["doi"] not in landing_text
        or dissertation["pdf"]["md5"] not in landing_text
    ):
        raise ValueError("Schützenberger repository metadata changed")
    reader = PdfReader(dissertation_path, strict=False)
    if len(reader.pages) != int(dissertation["expected_page_count"]) or len(
        reader.attachments
    ) != int(dissertation["expected_attachment_count"]):
        raise ValueError("Schützenberger dissertation inventory changed")
    drews_pages = [" ".join((page.extract_text() or "").split()) for page in reader.pages]
    drews_terms = _verify_term_inventory(drews_pages, dissertation, "Schützenberger")
    drews_text = " ".join(drews_pages)
    required_drews_phrases = (
        "Mi4: 20/13",
        "The imaging acquisition rate was 11.8 Hz for all experiments",
        "Relative ﬂuorescence changes (∆F/F ) from raw calcium traces",
        "tonic Mi4, Mi9, and Tm9 showed similar tuning as L1–5 and again little "
        "surround-dependency",
    )
    if any(phrase not in drews_text for phrase in required_drews_phrases):
        raise ValueError("Drews Mi4 experiment provenance text changed")
    article_paths = {
        name: _verify_file(root, spec)
        for name, spec in {
            "open_article_pdf": drews["article"]["open_article_pdf"],
            "supplement_pdf": drews["article"]["supplement_pdf"],
        }.items()
    }
    article_reader = PdfReader(article_paths["open_article_pdf"], strict=False)
    supplement_reader = PdfReader(article_paths["supplement_pdf"], strict=False)
    article_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in article_reader.pages
    )
    supplement_text = " ".join(
        " ".join((page.extract_text() or "").split())
        for page in supplement_reader.pages
    )
    if (
        drews["article"]["title"] not in article_text
        or "Mi4 in N: 20/13" not in article_text
        or "Mi4-GCaMP6f" not in supplement_text
    ):
        raise ValueError("Drews article or supplement evidence changed")

    repository = drews["repository"]
    readme_path = _verify_file(root, repository["readme"])
    payload_path = _verify_file(root, repository["figure3_payload"])
    readme = readme_path.read_text(encoding="utf-8")
    for phrase in (
        "serialized pandas-Dataframe for the data in Figure 3",
        '"cell_type"   : L1, L2, L3, L4, L5, Mi1, Tm3, Mi4',
        '"name"        : individual name given to each ROI',
        "zero defined as the time point when the stimulus started moving",
    ):
        if phrase not in readme:
            raise ValueError("Drews repository README contract changed")
    frame, globals_used = _load_restricted(payload_path)
    expected = drews["expected_payload"]
    if list(frame.shape) != expected["full_shape"]:
        raise ValueError("Drews Figure 3 full shape changed")
    if frame.index.names != expected["index_names"] or list(frame.columns) != expected["columns"]:
        raise ValueError("Drews Figure 3 axes changed")
    mi4 = frame.xs("Mi4", level="cell_type", drop_level=False)
    if list(mi4.shape) != expected["Mi4_shape"]:
        raise ValueError("Drews Mi4 shape changed")
    names = sorted(str(value) for value in mi4.index.get_level_values("name").unique())
    matches = [re.fullmatch(r"(fly\d+)_acq\d+_ROI\d+", name) for name in names]
    if any(match is None for match in matches):
        raise ValueError("Drews Mi4 ROI naming convention changed")
    fly_labels = sorted({match.group(1) for match in matches if match is not None})
    foreground = sorted(
        float(value) for value in mi4.index.get_level_values("fg_contrast").unique()
    )
    background = sorted(
        float(value) for value in mi4.index.get_level_values("bg_contrast").unique()
    )
    trial_labels = sorted(int(value) for value in mi4.index.get_level_values("trial").unique())
    conditions = mi4.reset_index()[["fg_contrast", "bg_contrast"]].drop_duplicates()
    time = np.sort(mi4["time"].unique())
    signal = mi4["signal"].to_numpy(dtype=float)
    checks = (
        len(names) == int(expected["Mi4_roi_count"]),
        len(fly_labels) == int(expected["Mi4_fly_count"]),
        len(conditions) == int(expected["condition_count"]),
        trial_labels == expected["trial_labels"],
        foreground == expected["foreground_contrasts"],
        background == expected["background_contrasts"],
        len(time) == int(expected["time_sample_count"]),
        math.isclose(
            float(np.median(np.diff(time))),
            float(expected["sample_interval_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        math.isclose(
            float(time.min()),
            float(expected["time_minimum_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        math.isclose(
            float(time.max()),
            float(expected["time_maximum_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        int(np.isnan(signal).sum()) == int(expected["missing_signal_count"]),
        int(np.isfinite(signal).sum()) == int(expected["finite_signal_count"]),
        not np.isinf(signal).any(),
    )
    if not all(checks):
        raise ValueError("Drews Mi4 payload inventory changed")

    sporar = config["sporar_2020"]
    metadata_path = _verify_file(root, sporar["metadata"])
    sporar_path = _verify_file(root, sporar["pdf"])
    metadata = metadata_path.read_text(encoding="utf-8")
    if (
        sporar["title"] not in metadata
        or "Sporar, Katja" not in metadata
        or sporar["doi"] not in metadata
    ):
        raise ValueError("Sporar official metadata changed")
    sporar_reader = PdfReader(sporar_path, strict=False)
    if len(sporar_reader.pages) != int(sporar["expected_page_count"]) or len(
        sporar_reader.attachments
    ) != int(sporar["expected_attachment_count"]):
        raise ValueError("Sporar dissertation inventory changed")
    sporar_pages = [
        " ".join((page.extract_text() or "").split()) for page in sporar_reader.pages
    ]
    sporar_terms = _verify_term_inventory(sporar_pages, sporar, "Sporar")
    sporar_text = " ".join(sporar_pages)
    required_sporar_phrases = (
        "silencing of C2 and C3 resulted in reduced behavior to regressive motion stimuli",
        "both C2 and C3 are promising candidates to eliminate the L2 baseline",
        "we could test t he role of GABAergic C2 and C3 neurons",
    )
    if any(phrase not in sporar_text for phrase in required_sporar_phrases):
        raise ValueError("Sporar C3 context boundary changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
            },
            "raw_snapshot_identity": {
                spec["path"]: _sha256(root / spec["path"])
                for spec in (
                    dissertation["landing_page"],
                    dissertation["pdf"],
                    drews["article"]["open_article_pdf"],
                    drews["article"]["supplement_pdf"],
                    repository["readme"],
                    repository["figure3_payload"],
                    sporar["metadata"],
                    sporar["pdf"],
                )
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "Drews_2020": {
            "article_doi": drews["article"]["doi"],
            "dissertation_doi": dissertation["doi"],
            "dissertation_page_count": len(reader.pages),
            "dissertation_term_evidence": drews_terms,
            "repository": {
                "url": repository["url"],
                "head_commit": repository["head_commit"],
                "data_commit": repository["data_commit"],
                "commit_count": repository["commit_count"],
                "branch": repository["branch"],
                "tracked_file_count": repository["tracked_file_count"],
            },
            "pickle_security": {
                "restricted_unpickler_used": True,
                "globals_used": globals_used,
                "arbitrary_repository_code_executed": False,
            },
            "Mi4_payload": {
                "shape": list(mi4.shape),
                "ROI_count": len(names),
                "pseudonymous_fly_count": len(fly_labels),
                "ROI_labels": names,
                "pseudonymous_fly_labels": fly_labels,
                "condition_count": len(conditions),
                "foreground_contrasts": foreground,
                "background_contrasts": background,
                "trial_labels": trial_labels,
                "time_sample_count": len(time),
                "time_minimum_seconds": float(time.min()),
                "time_maximum_seconds": float(time.max()),
                "sample_interval_seconds": float(np.median(np.diff(time))),
                "effective_sample_rate_hz": float(1.0 / np.median(np.diff(time))),
                "missing_signal_count": int(np.isnan(signal).sum()),
                "finite_signal_count": int(np.isfinite(signal).sum()),
                "response_modality": "two_photon_GCaMP6f_calcium_imaging",
                "response_unit": "deltaF_over_F",
                "individual_ROI_trial_numeric_time_series_verified": True,
                "paper_reported_20_cells_13_flies_reconstructed": True,
                "stable_pseudonymous_fly_axis_available": True,
                "recording_to_MaleCNS_body_crosswalk_found": False,
            },
            "Mi4_direct_neural_recording_verified": True,
            "C3_direct_neural_recording_verified": False,
            "experimental_membrane_voltage_verified": False,
            "classification": "public_individual_Mi4_GCaMP6f_time_series",
            "authorize_Mi4_C3_voltage_transfer": False,
            "authorize_T4_source_dynamics_fit": False,
        },
        "Sporar_2020": {
            "doi": sporar["doi"],
            "page_count": len(sporar_reader.pages),
            "term_evidence": sporar_terms,
            "directly_recorded_neural_activity_sources_in_C3_context": [],
            "C3_direct_neural_recording_verified": False,
            "Mi4_direct_neural_recording_verified": False,
            "classification": "C3_anatomy_prior_intervention_and_future_experiment_context_only",
            "authorize_Mi4_C3_source_dynamics_transfer": False,
        },
        "independent_Mi4_numeric_calcium_dynamics_verified": True,
        "independent_C3_numeric_dynamics_verified": False,
        "independent_Mi4_C3_numeric_membrane_voltage_verified": False,
        "authorize_T4_source_dynamics_fit": False,
        "advance_to_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": "Mi4_calcium_is_not_Mi4_C3_numeric_membrane_voltage_and_C3_is_unmeasured",
        "boundary": config["boundary"],
    }
