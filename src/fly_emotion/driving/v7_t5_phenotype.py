from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.error
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_t5_conductance_audit import (
    _clone_fixed_repository,
    _load_cell,
    _segments,
    _vector,
    audit_t5_conductance_repository,
)
from fly_emotion.driving.v7_t5_data_audit import _fetch, _plain

CONFIG = Path("configs/driving-v7-t5-phenotype.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_t5_phenotype.py")


def direction_contrast(direction_0_peak: float, direction_1_peak: float) -> float:
    denominator = max(abs(direction_0_peak), abs(direction_1_peak))
    if denominator == 0:
        return 0.0
    return float((direction_1_peak - direction_0_peak) / denominator)


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "pair_count": int(array.size),
        "positive_count": int(np.count_nonzero(array > 0)),
        "zero_count": int(np.count_nonzero(array == 0)),
        "negative_count": int(np.count_nonzero(array < 0)),
        "positive_fraction": float(np.mean(array > 0)),
        "median_contrast": float(np.median(array)),
        "lower_quartile_contrast": float(np.quantile(array, 0.25)),
        "upper_quartile_contrast": float(np.quantile(array, 0.75)),
    }


def extract_moving_bar_pairs(source: Path, recorded_cells: int) -> list[dict]:
    pairs = []
    for cell_id in range(1, recorded_cells + 1):
        data, protocol = _load_cell(source, cell_id, "all")
        values, starts, stops = _segments(data, "mb")
        widths = _vector(protocol.width_mb).astype(int)
        durations = _vector(protocol.duration_mb).astype(int)
        directions = _vector(protocol.direction_mb).astype(int)
        if not (starts.size == widths.size == durations.size == directions.size):
            raise ValueError(f"moving-bar metadata mismatch for cell {cell_id}")
        grouped: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        for start, stop, width, duration, direction in zip(
            starts, stops, widths, durations, directions, strict=True
        ):
            if int(direction) not in {0, 1}:
                raise ValueError(f"unexpected moving-bar direction code for cell {cell_id}")
            key = (int(width), int(duration))
            if int(direction) in grouped[key]:
                raise ValueError(f"duplicate moving-bar direction condition for cell {cell_id}")
            grouped[key][int(direction)] = float(np.max(values[start - 1 : stop]))
        for (width, duration), peaks in sorted(grouped.items()):
            if set(peaks) != {0, 1}:
                raise ValueError(f"unpaired moving-bar condition for cell {cell_id}")
            pairs.append(
                {
                    "cell_id": cell_id,
                    "width_pixels": width,
                    "step_duration_milliseconds": duration,
                    "direction_0_peak_millivolts": peaks[0],
                    "direction_1_peak_millivolts": peaks[1],
                    "direction_1_minus_0_normalized_contrast": direction_contrast(
                        peaks[0], peaks[1]
                    ),
                }
            )
    return pairs


def _group_summaries(pairs: list[dict], field: str) -> dict:
    output = {}
    for value in sorted({item[field] for item in pairs}):
        contrasts = [
            item["direction_1_minus_0_normalized_contrast"]
            for item in pairs
            if item[field] == value
        ]
        output[str(value)] = _summary(contrasts)
    return output


def _fetch_available_paper(sources: list[dict]) -> tuple[bytes, int, str]:
    errors = []
    for source in sources:
        try:
            raw, status = _fetch(source["url"])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            errors.append(f"{source['url']}: {type(error).__name__}")
            continue
        return raw, status, source["url"]
    raise ValueError("no paper source was available: " + "; ".join(errors))


def evaluate_v7_t5_phenotype(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    repository_config_path = Path(config["source"]["repository_config"])
    repository_evidence_path = Path(config["source"]["repository_evidence"])
    repository_config = yaml.safe_load((root / repository_config_path).read_text(encoding="utf-8"))
    repository_evidence = json.loads((root / repository_evidence_path).read_text(encoding="utf-8"))
    if not config["exploratory"] or config["advance_allowed"]:
        raise ValueError("T5 phenotype extraction must remain exploratory and non-advancing")
    if repository_evidence["advance_to_T5_fit"]:
        raise ValueError("T5 source evidence unexpectedly allows fitting")
    article_raw, article_status, article_url = _fetch_available_paper(
        config["source"]["paper_sources"]
    )
    article_text = _plain(article_raw.decode("utf-8"))
    normalized_article = article_text.replace("−", "-").replace("–", "-")
    for phrase in config["source"]["required_paper_phrases"]:
        normalized_phrase = str(phrase).replace("−", "-").replace("–", "-")
        if normalized_phrase.lower() not in normalized_article.lower():
            raise ValueError(f"paper is missing required phenotype phrase: {phrase}")
    evidence_claim_sha256 = hashlib.sha256(
        "\n".join(config["source"]["required_paper_phrases"]).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="autodrive-v7-t5-phenotype-") as temporary:
        source = Path(temporary) / "repository"
        _clone_fixed_repository(repository_config, source)
        verified = audit_t5_conductance_repository(source, repository_config)
        if (
            verified["repository"]["manifest_sha256"]
            != repository_evidence["repository"]["manifest_sha256"]
        ):
            raise ValueError("T5 phenotype repository differs from audited evidence")
        pairs = extract_moving_bar_pairs(source, int(repository_config["paper"]["recorded_cells"]))
    contrasts = [item["direction_1_minus_0_normalized_contrast"] for item in pairs]
    rng = np.random.default_rng(int(config["analysis"]["permutation_seed"]))
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(contrasts), replace=True)
    permuted = (np.asarray(contrasts) * signs).tolist()
    cell_medians = {
        str(cell_id): float(
            np.median(
                [
                    item["direction_1_minus_0_normalized_contrast"]
                    for item in pairs
                    if item["cell_id"] == cell_id
                ]
            )
        )
        for cell_id in range(1, int(repository_config["paper"]["recorded_cells"]) + 1)
    }
    return {
        "protocol": {
            "name": config["name"],
            "exploratory": True,
            "advance_allowed": False,
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(repository_config_path): _sha256(root / repository_config_path),
                str(repository_evidence_path): _sha256(root / repository_evidence_path),
            },
            "parameter_fitting": False,
            "model_simulation": False,
            "raw_files_committed": False,
            "temporary_clone_deleted_after_audit": True,
        },
        "paper_evidence": {
            "url": article_url,
            "http_status": article_status,
            "validated_claims_sha256": evidence_claim_sha256,
            "reported_cells": 17,
            "reports_all_cells_have_PD_and_ND": True,
            "reported_DSI_definition": "(PDmax-NDmax)/PDmax",
            "raw_acquisition_kilohertz": 20,
            "raw_low_pass_kilohertz": 10,
            "responses_baseline_subtracted": True,
        },
        "repository_evidence": {
            "commit": verified["repository"]["commit"],
            "manifest_sha256": verified["repository"]["manifest_sha256"],
            "recorded_cells": len(verified["cells"]),
        },
        "moving_bar_pairs": pairs,
        "summary": {
            "all_pairs": _summary(contrasts),
            "by_width_pixels": _group_summaries(pairs, "width_pixels"),
            "by_step_duration_milliseconds": _group_summaries(pairs, "step_duration_milliseconds"),
            "cell_median_contrast": cell_medians,
            "cells_with_positive_median": int(sum(value > 0 for value in cell_medians.values())),
        },
        "negative_controls": {
            "swap_direction_codes_median_contrast": float(np.median(-np.asarray(contrasts))),
            "swap_negation_maximum_error": float(
                np.max(np.abs(-np.asarray(contrasts) + np.asarray(contrasts)))
            ),
            "deterministic_within_pair_label_permutation": _summary(permuted),
        },
        "label_boundary": {
            **config["interpretation_boundary"],
            "direction_1_has_larger_peak_pairs": int(sum(value > 0 for value in contrasts)),
            "direction_1_has_larger_peak_fraction": float(np.mean(np.asarray(contrasts) > 0)),
            "biological_PD_code_assigned": None,
            "reason": (
                "The paper states that traces were aligned to each cell's PD-ND axis, but the "
                "audited repository does not label numeric direction codes as PD or ND. Response "
                "magnitude is therefore not used to assign the biological label."
            ),
        },
        "limitations": [
            "This report describes measured T5 phenotype and does not score a v7 model.",
            (
                "Direction-code asymmetry is not biological direction correctness without "
                "a source label map."
            ),
            "The moving-bar cells are the same cells whose width-2 flashes define model fits.",
            "The processed 2.5/5-ms traces are not the raw 20-kHz acquisition.",
            "No independent-cell or untouched final test is created by this audit.",
        ],
        "advance_to_T5_fit": False,
        "advance_to_visual_gate": False,
        "advance_to_central_complex": False,
    }
