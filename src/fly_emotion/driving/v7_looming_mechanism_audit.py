from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-looming-mechanism-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_looming_mechanism_audit.py")


def _plain(raw: bytes) -> str:
    text = unescape(raw.decode("utf-8"))
    output: list[str] = []
    inside = False
    for character in text:
        if character == "<":
            inside = True
        elif character == ">":
            inside = False
            output.append(" ")
        elif not inside:
            output.append(character)
    return " ".join("".join(output).split())


def _fetch(url: str) -> tuple[bytes, int]:
    request = urllib.request.Request(url, headers={"User-Agent": "AutoDrive-Fly-v7-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read(), int(response.status)
    except urllib.error.HTTPError as error:
        raise ValueError(f"literature source unavailable: HTTP {error.code} {url}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"literature source unavailable: {url}") from error


def _current_battery_axes(stage1: dict) -> set[str]:
    families = set(stage1["families"])
    axes = set()
    if "looming" in families:
        axes.update({"receptive_field_centered_dark_looming", "receding_control"})
    if "static" in families:
        axes.add("luminance_matched_motion_free_darkening")
    if "translation" in families:
        axes.add("wide_field_translation_control")
    return axes


def evaluate_v7_looming_mechanism_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    stage1_path = Path(config["stage1_split_config"])
    scoring_path = Path(config["stage1_scoring_config"])
    structure_path = Path(config["structure_evidence"])
    stage1 = yaml.safe_load((root / stage1_path).read_text(encoding="utf-8"))
    scoring = yaml.safe_load((root / scoring_path).read_text(encoding="utf-8"))
    structure = json.loads((root / structure_path).read_text(encoding="utf-8"))
    if structure["advance_to_model_change"]:
        raise ValueError("structure evidence unexpectedly authorizes model changes")
    if scoring["aggregation"]["looming_comparator"] != (
        "elementwise_max_of_receding_and_same_size_static"
    ):
        raise ValueError("current shared looming score changed")
    current_axes = _current_battery_axes(stage1)
    sources = {}
    for target, source in config["sources"].items():
        raw, status = _fetch(source["url"])
        text = _plain(raw)
        missing = [
            phrase for phrase in source["required_phrases"] if phrase.lower() not in text.lower()
        ]
        if missing:
            raise ValueError(f"{target} source is missing required phrases: {missing}")
        sources[target] = {
            "paper_doi": source["paper_doi"],
            "pubmed_id": source["pubmed_id"],
            "url": source["url"],
            "http_status": status,
            "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "required_phrases_verified": source["required_phrases"],
        }
    coverage = {}
    for target, contract in config["target_contracts"].items():
        required = list(contract["required_stimulus_axes"])
        present = [axis for axis in required if axis in current_axes]
        missing = [axis for axis in required if axis not in current_axes]
        coverage[target] = {
            "output_role": contract["output_role"],
            "required_stimulus_axes": required,
            "covered_by_current_stage1_battery": present,
            "missing_from_current_stage1_battery": missing,
            "coverage_fraction": len(present) / len(required),
            "current_shared_looming_score_is_sufficient": contract[
                "current_shared_looming_score_is_sufficient"
            ],
            "ready_for_typed_development_fit": False,
        }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(stage1_path): _sha256(root / stage1_path),
                str(scoring_path): _sha256(root / scoring_path),
                str(structure_path): _sha256(root / structure_path),
            },
            "literature_audit_only": True,
            "parameter_fitting": False,
            "neural_evaluation_performed": False,
            "runtime_modified": False,
        },
        "sources": sources,
        "current_stage1_axes": sorted(current_axes),
        "target_contracts": coverage,
        "mechanism_boundaries": {
            "LPLC1": (
                "near-collision slowing depends on object position, motion direction and "
                "relative background motion; generic radial expansion is insufficient"
            ),
            "LPLC2": (
                "local outward motion with inward-motion inhibition rejects receding, "
                "motion-free darkening and wide-field translation"
            ),
            "LC4": (
                "LC4 contributes an angular-velocity component downstream at GF; it is not "
                "interchangeable with the LPLC2 terminal-size component"
            ),
        },
        "next_stimulus_requirements": {
            target: item["missing_from_current_stage1_battery"]
            for target, item in coverage.items()
        },
        "boundary": config["boundary"],
        "limitations": [
            "Literature statements define stimulus semantics, not MaleCNS parameter values.",
            "LPLC1 and LPLC2 evidence is calcium imaging, not absolute membrane voltage.",
            "The LC4 source available here is a PubMed abstract and downstream GF model summary.",
            "No current v7 target response is reclassified or rescored by this audit.",
        ],
        "advance_to_typed_looming_model": False,
        "advance_to_validation": False,
        "advance_to_central_complex": False,
    }
