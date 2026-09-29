"""Evaluate the preregistered MaleCNS Tm4 column-offset replication."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_malecns_synapse_column_audit import (
    _official_synapse_count_candidate,
)
from fly_emotion.driving.v7_malecns_tm4_synapse_column_boundary_audit import (
    _hex_distance,
)

CONFIG = Path("configs/driving-v7-malecns-tm4-offset-replication.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_tm4_offset_replication.py"
)


def _git_sha256(root: Path, revision: str, path: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _one_sided_exact_sign_pvalue(improved: int, worsened: int) -> float:
    """Return P[X >= improved] for X~Binomial(improved+worsened, 0.5)."""
    discordant = improved + worsened
    if min(improved, worsened) < 0 or discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, value) for value in range(improved, discordant + 1)
    ) / (2**discordant)


def _score_candidate(
    candidate_hex: list[int] | None, native_hex: list[int], correction: list[int]
) -> dict:
    corrected = (
        [
            int(candidate_hex[0]) + int(correction[0]),
            int(candidate_hex[1]) + int(correction[1]),
        ]
        if candidate_hex is not None
        else None
    )
    return {
        "hex": corrected,
        "exact_native_match": corrected == native_hex,
        "hex_distance_to_native": (
            _hex_distance(corrected, native_hex) if corrected is not None else None
        ),
    }


def _summarize_records(records: list[dict], sample_count: int, alpha: float) -> dict:
    unique_count = sum(record["unique_candidate"] for record in records)
    uncorrected_count = sum(
        record["uncorrected"]["exact_native_match"] for record in records
    )
    corrected_count = sum(
        record["corrected"]["exact_native_match"] for record in records
    )
    improved = sum(
        not record["uncorrected"]["exact_native_match"]
        and record["corrected"]["exact_native_match"]
        for record in records
    )
    worsened = sum(
        record["uncorrected"]["exact_native_match"]
        and not record["corrected"]["exact_native_match"]
        for record in records
    )
    pvalue = _one_sided_exact_sign_pvalue(improved, worsened)
    gates = {
        "all_sampled_bodies_retrieved": len(records) == sample_count,
        "unique_synapse_count_candidate_for_every_body": unique_count == sample_count,
        "corrected_exact_count_greater_than_uncorrected_exact_count": (
            corrected_count > uncorrected_count
        ),
        "no_correct_to_incorrect_pairs": worsened == 0,
        "paired_one_sided_exact_sign_test_passed": pvalue <= alpha,
        "no_distance_tolerance_used": True,
    }
    return {
        "sample_count": len(records),
        "unique_candidate_count": unique_count,
        "uncorrected_exact_count": uncorrected_count,
        "corrected_exact_count": corrected_count,
        "paired_exact_outcomes": {
            "incorrect_to_correct": improved,
            "correct_to_incorrect": worsened,
            "discordant_count": improved + worsened,
            "one_sided_exact_sign_test_pvalue": pvalue,
            "alpha": alpha,
        },
        "gates": gates,
        "replication_gate_passed": all(gates.values()),
    }


def evaluate_v7_malecns_tm4_offset_replication(root: Path) -> dict:
    audit = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    prereg_path = Path(audit["preregistration_evidence"])
    prereg_sha256 = audit["preregistration_sha256"]
    if _sha256(root / prereg_path) != prereg_sha256:
        raise ValueError("Tm4 preregistration artifact hash changed")
    revision = audit["preregistration_commit"]
    if _git_sha256(root, revision, str(prereg_path)) != prereg_sha256:
        raise ValueError("Tm4 preregistration was not frozen at declared commit")
    prereg = json.loads((root / prereg_path).read_text(encoding="utf-8"))
    if not prereg["replication_protocol_frozen"]:
        raise ValueError("Tm4 replication protocol is not frozen")
    if prereg["protocol"]["replication_outputs_observed"]:
        raise ValueError("Tm4 preregistration already contains replication output")
    if prereg["replication_gate"] != audit["replication_gate"]:
        raise ValueError("Tm4 replication gates changed after preregistration")
    retrieval_path = Path(audit["retrieval_implementation"])
    retrieval_revision = audit["retrieval_implementation_commit"]
    retrieval_sha256 = audit["retrieval_implementation_sha256"]
    if _sha256(root / retrieval_path) != retrieval_sha256:
        raise ValueError("Tm4 retrieval implementation hash changed")
    if _git_sha256(root, retrieval_revision, str(retrieval_path)) != retrieval_sha256:
        raise ValueError("Tm4 retrieval implementation was not frozen at commit")

    snapshot_spec = audit["replication_snapshot"]
    if not snapshot_spec["observed"]:
        raise ValueError("Tm4 replication snapshot has not been observed")
    if snapshot_spec["bytes"] is None or snapshot_spec["sha256"] is None:
        raise ValueError("Tm4 replication snapshot identity is incomplete")
    snapshot_path = root / snapshot_spec["path"]
    if (
        snapshot_path.stat().st_size != int(snapshot_spec["bytes"])
        or _sha256(snapshot_path) != snapshot_spec["sha256"]
    ):
        raise ValueError("Tm4 replication snapshot changed")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot["protocol"]["preregistration_sha256"] != prereg_sha256:
        raise ValueError("Tm4 snapshot references another preregistration")
    if snapshot["protocol"]["retrieval_implementation_sha256"] != retrieval_sha256:
        raise ValueError("Tm4 snapshot references another retrieval implementation")
    expected_ids = [int(value) for value in prereg["replication_sample_body_ids"]]
    rows = snapshot["rows"]
    observed_ids = [int(row["body_id"]) for row in rows]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(expected_ids):
        raise ValueError("Tm4 replication body identities or order changed")
    if len(rows) != int(snapshot["protocol"]["body_count"]):
        raise ValueError("Tm4 replication body count changed")

    annotation_spec = audit["annotations"]
    annotation_path = root / annotation_spec["path"]
    if (
        annotation_path.stat().st_size != int(annotation_spec["bytes"])
        or _sha256(annotation_path) != annotation_spec["sha256"]
    ):
        raise ValueError("Tm4 annotation snapshot changed")
    annotations = feather.read_table(
        annotation_path,
        columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()
    selected_annotations = annotations.loc[annotations["bodyId"].isin(expected_ids)]
    if (
        len(selected_annotations) != len(expected_ids)
        or not selected_annotations["bodyId"].is_unique
        or selected_annotations[["assignedOlHex1", "assignedOlHex2"]]
        .isna()
        .any(axis=None)
    ):
        raise ValueError("Tm4 replication native annotations are incomplete")
    annotations = selected_annotations.set_index("bodyId")
    pin_map = {
        int(column_id): matches
        for column_id, matches in snapshot["column_pin_map"].items()
    }
    rules = prereg["candidate_rules"]
    if rules != audit["candidate_rules"]:
        raise ValueError("Tm4 correction rules changed after preregistration")

    records = []
    for row in rows:
        body_id = int(row["body_id"])
        annotation = annotations.loc[body_id]
        if annotation["type"] != "Tm4" or annotation["somaSide"] != "R":
            raise ValueError("Tm4 replication annotation identity changed")
        native_hex = [
            int(annotation["assignedOlHex1"]),
            int(annotation["assignedOlHex2"]),
        ]
        candidate = _official_synapse_count_candidate(row, pin_map, "R")
        raw_hex = candidate["recovered_hex"] if candidate["unique_mode"] else None
        uncorrected = _score_candidate(raw_hex, native_hex, rules["uncorrected"])
        corrected = _score_candidate(
            raw_hex, native_hex, rules["discovery_offset_mode_correction"]
        )
        records.append(
            {
                "body_id": body_id,
                "native_hex": native_hex,
                "unique_candidate": candidate["unique_mode"],
                "raw_candidate_hex": raw_hex,
                "uncorrected": uncorrected,
                "corrected": corrected,
            }
        )

    gate = prereg["replication_gate"]
    if not gate["no_distance_tolerance_acceptance_gate"]:
        raise ValueError("Tm4 preregistration distance-tolerance ban changed")
    summary = _summarize_records(
        records,
        len(expected_ids),
        float(gate["paired_one_sided_exact_sign_test_alpha"]),
    )
    replication_passed = summary["replication_gate_passed"]
    return {
        "protocol": {
            "name": audit["name"],
            "observed_on": audit["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(prereg_path): prereg_sha256,
                str(retrieval_path): retrieval_sha256,
                snapshot_spec["path"]: snapshot_spec["sha256"],
                annotation_spec["path"]: annotation_spec["sha256"],
            },
            "preregistration_commit": revision,
            "preregistration_sha256": prereg_sha256,
            "discovery_body_ids_excluded": True,
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "replication_evaluated": True,
        **summary,
        "uncorrected_hex_distance_counts": dict(
            sorted(
                Counter(
                    str(record["uncorrected"]["hex_distance_to_native"])
                    for record in records
                    if record["uncorrected"]["hex_distance_to_native"] is not None
                ).items()
            )
        ),
        "corrected_hex_distance_counts": dict(
            sorted(
                Counter(
                    str(record["corrected"]["hex_distance_to_native"])
                    for record in records
                    if record["corrected"]["hex_distance_to_native"] is not None
                ).items()
            )
        ),
        "records": records,
        "authorize_post_hoc_hex_distance_tolerance": False,
        "authorize_left_Tm4_coordinate_writeback": False,
        "authorize_source_mapping_gate_change": False,
        "advance_to_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "offset_replication_passed_but_left_writeback_requires_independent_validation"
            if replication_passed
            else "preregistered_Tm4_offset_replication_gate_failed"
        ),
        "boundary": prereg["boundary"],
    }
