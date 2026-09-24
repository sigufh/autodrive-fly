"""Bound official synapse-count coordinates for one-sided Tm4 gaps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_malecns_synapse_column_audit import (
    _official_synapse_count_candidate,
)

CONFIG = Path(
    "configs/driving-v7-malecns-tm4-synapse-column-boundary-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_malecns_tm4_synapse_column_boundary_audit.py"
)


def _spaced_body_ids(frame, count: int, sort_columns: list[str]) -> list[int]:
    ordered = frame.sort_values(sort_columns)
    indices = sorted(
        {round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)}
    )
    if len(indices) != count:
        raise ValueError("Tm4 deterministic sample contains duplicate indices")
    return ordered.iloc[indices]["bodyId"].astype(int).tolist()


def _hex_distance(first: list[int], second: list[int]) -> int:
    delta_q = int(first[0]) - int(second[0])
    delta_r = int(first[1]) - int(second[1])
    return max(abs(delta_q), abs(delta_r), abs(delta_q + delta_r))


def evaluate_v7_malecns_tm4_synapse_column_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    annotation_path = Path(config["annotations"])
    snapshot_spec = config["sample_snapshot"]
    snapshot_path = root / snapshot_spec["path"]
    if (
        snapshot_path.stat().st_size != int(snapshot_spec["bytes"])
        or _sha256(snapshot_path) != snapshot_spec["sha256"]
    ):
        raise ValueError("bounded Tm4 synapse-column snapshot changed")
    official_path = Path(config["official_implementation"])
    query_path = Path(config["official_query"])
    retrieval_path = Path(config["retrieval_implementation"])
    official_text = (root / official_path).read_text(encoding="utf-8")
    query_text = (root / query_path).read_text(encoding="utf-8")
    if (
        "method:str='synapse_count'" not in official_text
        or "sort_values(['ROI', 'synapse_count'], ascending=False)"
        not in official_text
        or "count(*) as synapse_count" not in query_text
    ):
        raise ValueError("official MaleCNS synapse-count semantics changed")

    annotations = feather.read_table(
        root / annotation_path,
        columns=[
            "bodyId",
            "type",
            "somaSide",
            "assignedOlHex1",
            "assignedOlHex2",
        ],
    ).to_pandas()
    tm4 = annotations.loc[annotations["type"].eq("Tm4")].copy()
    expected = config["expected"]
    body_count_by_side = {
        side: int(tm4["somaSide"].eq(side).sum()) for side in ("L", "R")
    }
    native_count_by_side = {
        side: int(
            (
                tm4["somaSide"].eq(side)
                & tm4["assignedOlHex1"].notna()
                & tm4["assignedOlHex2"].notna()
            ).sum()
        )
        for side in ("L", "R")
    }
    if body_count_by_side != expected["body_count_by_side"]:
        raise ValueError("Tm4 body counts by side changed")
    if native_count_by_side != expected["native_count_by_side"]:
        raise ValueError("Tm4 native coordinate counts by side changed")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = snapshot["rows"]
    pin_map = {
        int(column_id): values
        for column_id, values in snapshot["column_pin_map"].items()
    }
    sample_count = int(expected["sample_count_by_side"])
    right_native = tm4.loc[
        tm4["somaSide"].eq("R") & tm4["assignedOlHex1"].notna()
    ]
    left_missing = tm4.loc[tm4["somaSide"].eq("L")]
    expected_ids = {
        "R": _spaced_body_ids(
            right_native,
            sample_count,
            ["assignedOlHex1", "assignedOlHex2", "bodyId"],
        ),
        "L": _spaced_body_ids(left_missing, sample_count, ["bodyId"]),
    }
    sampled_ids = {
        side: [int(row["body_id"]) for row in rows if row["side"] == side]
        for side in ("L", "R")
    }
    if sampled_ids != expected_ids:
        raise ValueError("Tm4 bounded sample identities changed")

    results = {}
    for side in ("R", "L"):
        records = []
        for row in rows:
            if row["side"] != side:
                continue
            candidate = _official_synapse_count_candidate(row, pin_map, side)
            native = row["native_hex"]
            records.append(
                {
                    "body_id": int(row["body_id"]),
                    "native_hex": native,
                    "unique_candidate": candidate["unique_mode"],
                    "candidate_hex": candidate["recovered_hex"],
                    "candidate_matches_native": bool(
                        native is not None
                        and candidate["unique_mode"]
                        and candidate["recovered_hex"] == native
                    ),
                    "candidate_to_native_hex_distance": (
                        _hex_distance(candidate["recovered_hex"], native)
                        if native is not None and candidate["unique_mode"]
                        else None
                    ),
                }
            )
        unique_count = sum(item["unique_candidate"] for item in records)
        exact_count = sum(item["candidate_matches_native"] for item in records)
        results[side] = {
            "sample_count": len(records),
            "sample_body_ids": sampled_ids[side],
            "native_coordinate_available": side == "R",
            "unique_candidate_count": unique_count,
            "unique_candidate_fraction": unique_count / len(records),
            "exact_native_match_count": exact_count if side == "R" else None,
            "exact_native_match_fraction": (
                exact_count / unique_count if side == "R" and unique_count else None
            ),
            "candidate_to_native_hex_distance_counts": (
                {
                    str(distance): count
                    for distance, count in sorted(
                        Counter(
                            item["candidate_to_native_hex_distance"]
                            for item in records
                            if item["candidate_to_native_hex_distance"] is not None
                        ).items()
                    )
                }
                if side == "R"
                else None
            ),
            "distinct_candidate_hex_count": len(
                {
                    tuple(item["candidate_hex"])
                    for item in records
                    if item["unique_candidate"]
                }
            ),
            "records": records,
        }
    if results["R"]["unique_candidate_count"] != int(
        expected["right_unique_candidate_count"]
    ) or results["R"]["exact_native_match_count"] != int(
        expected["right_exact_candidate_count"]
    ):
        raise ValueError("right Tm4 synapse-count validation changed")
    if results["L"]["unique_candidate_count"] != int(
        expected["left_unique_candidate_count"]
    ):
        raise ValueError("left Tm4 synapse-count availability changed")
    expected_distances = {
        str(distance): int(count)
        for distance, count in expected["right_hex_distance_counts"].items()
    }
    if results["R"]["candidate_to_native_hex_distance_counts"] != expected_distances:
        raise ValueError("right Tm4 hex-distance distribution changed")
    if results["L"]["distinct_candidate_hex_count"] != int(
        expected["left_distinct_candidate_count"]
    ):
        raise ValueError("left Tm4 candidate collision count changed")

    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(annotation_path): _sha256(root / annotation_path),
                str(snapshot_spec["path"]): _sha256(snapshot_path),
                str(official_path): _sha256(root / official_path),
                str(query_path): _sha256(root / query_path),
                str(retrieval_path): _sha256(root / retrieval_path),
            },
            "parameter_fit": False,
            "target_activity_injection": False,
            "runtime_modified": False,
        },
        "Tm4_annotation_coverage": {
            "body_count_by_side": body_count_by_side,
            "native_count_by_side": native_count_by_side,
            "strictly_one_sided_native_coverage": True,
        },
        "sample_results": results,
        "right_native_validation_exact_fraction": results["R"][
            "exact_native_match_fraction"
        ],
        "left_missing_unique_candidate_fraction": results["L"][
            "unique_candidate_fraction"
        ],
        "right_native_candidate_hex_distance_counts": results["R"][
            "candidate_to_native_hex_distance_counts"
        ],
        "right_native_candidate_within_one_hex_fraction": (
            (
                results["R"]["candidate_to_native_hex_distance_counts"]["0"]
                + results["R"]["candidate_to_native_hex_distance_counts"]["1"]
            )
            / results["R"]["sample_count"]
        ),
        "left_missing_distinct_candidate_hex_count": results["L"][
            "distinct_candidate_hex_count"
        ],
        "authorize_post_hoc_hex_distance_tolerance": False,
        "official_synapse_count_candidate_is_native_equivalent_for_Tm4": False,
        "authorize_left_Tm4_coordinate_writeback": False,
        "authorize_source_mapping_gate_change": False,
        "authorize_T4_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_runtime_integration": False,
        "stop_reason": (
            "right_native_validation_exact_fraction_is_28_of_48_so_left_unique_"
            "candidates_are_not_authorized_as_native_writebacks"
        ),
        "boundary": config["boundary"],
    }
