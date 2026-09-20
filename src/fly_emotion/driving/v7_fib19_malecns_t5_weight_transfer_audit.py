"""Compare historical FIB19 and current MaleCNS T5 input-weight ratios."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256
from fly_emotion.driving.v7_kohn_portes_t5_ephys_audit import _verify_file
from fly_emotion.driving.v7_kohn_portes_tm_to_t5_model_audit import _git_blob_sha1

CONFIG = Path("configs/driving-v7-fib19-malecns-t5-weight-transfer-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_fib19_malecns_t5_weight_transfer_audit.py"
)


def _fib19_rows(path: Path, sources: list[str]) -> list[dict]:
    frame = pd.read_csv(
        path,
        header=None,
        skiprows=np.arange(0, 58, 3),
        nrows=40,
        usecols=np.arange(0, 30),
    )
    rows = []
    for index in range(0, frame.shape[0], 2):
        names = frame.loc[index, :]
        counts = frame.loc[index + 1, :]
        weights = {
            source: float(
                pd.to_numeric(counts[names.str.contains(source, regex=False)]).sum()
            )
            for source in [*sources, "CT1"]
        }
        total = sum(weights[source] for source in sources)
        rows.append(
            {
                "target_label": str(names.iloc[1]),
                "target_subtype": str(names.iloc[1]).split("-")[0],
                "source_weights": weights,
                "four_source_ratios": {
                    source: weights[source] / total for source in sources
                },
            }
        )
    return rows


def _mean_ratios(rows: list[dict], sources: list[str]) -> dict[str, float]:
    return {
        source: float(np.mean([row["four_source_ratios"][source] for row in rows]))
        for source in sources
    }


def _quantiles(rows: list[dict], sources: list[str]) -> dict[str, list[float]]:
    return {
        source: [
            float(value)
            for value in np.quantile(
                [row["four_source_ratios"][source] for row in rows],
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )
        ]
        for source in sources
    }


def evaluate_v7_fib19_malecns_t5_weight_transfer_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    expected = config["expected"]
    model_path = Path(config["author_model_evidence"])
    target_path = Path(config["malecns_target_input_evidence"])
    mapping_path = Path(config["source_mapping_evidence"])
    model = json.loads((root / model_path).read_text(encoding="utf-8"))
    target = json.loads((root / target_path).read_text(encoding="utf-8"))
    mapping = json.loads((root / mapping_path).read_text(encoding="utf-8"))
    source_table = _verify_file(root, config["fib19_source_table"])
    if _git_blob_sha1(source_table) != config["fib19_source_table"]["git_blob"]:
        raise ValueError("FIB19 T5 input table Git blob mismatch")

    sources = expected["sources"]
    fib19 = _fib19_rows(source_table, sources)
    subtype_counts = dict(sorted(Counter(row["target_subtype"] for row in fib19).items()))
    if len(fib19) != int(expected["fib19_T5_count"]):
        raise ValueError("FIB19 T5 count changed")
    if subtype_counts != expected["fib19_subtype_counts"]:
        raise ValueError("FIB19 T5 subtype balance changed")

    malecns = []
    zero_four_source = []
    for row in target["T5"]["targets"]:
        weights = {
            source: float(row["source_types"][source]["weight"])
            for source in [*sources, "CT1"]
        }
        total = sum(weights[source] for source in sources)
        if total <= 0.0:
            zero_four_source.append(int(row["body_id"]))
            continue
        malecns.append(
            {
                "body_id": int(row["body_id"]),
                "target_subtype": row["target_type"],
                "side": row["side"],
                "source_weights": weights,
                "four_source_ratios": {
                    source: weights[source] / total for source in sources
                },
            }
        )
    if len(target["T5"]["targets"]) != int(expected["malecns_T5_count"]):
        raise ValueError("MaleCNS T5 count changed")
    if len(malecns) != int(expected["malecns_four_source_nonzero_count"]):
        raise ValueError("MaleCNS nonzero four-source target count changed")
    if zero_four_source != expected["malecns_zero_four_source_body_ids"]:
        raise ValueError("MaleCNS zero four-source target IDs changed")

    fib19_mean = _mean_ratios(fib19, sources)
    malecns_mean = _mean_ratios(malecns, sources)
    for observed, declared, label in (
        (fib19_mean, expected["fib19_population_mean_four_source_ratios"], "FIB19"),
        (malecns_mean, expected["malecns_population_mean_four_source_ratios"], "MaleCNS"),
    ):
        if not all(
            np.isclose(observed[source], declared[source], rtol=0.0, atol=5e-15)
            for source in sources
        ):
            raise ValueError(f"{label} population mean ratio changed")
    differences = np.asarray([malecns_mean[source] - fib19_mean[source] for source in sources])
    if not np.isclose(
        np.mean(np.abs(differences)),
        expected["mean_absolute_population_ratio_difference"],
        rtol=0.0,
        atol=5e-15,
    ) or not np.isclose(
        np.max(np.abs(differences)),
        expected["maximum_population_ratio_difference"],
        rtol=0.0,
        atol=5e-15,
    ):
        raise ValueError("cross-connectome population ratio difference changed")

    fib19_quantiles = _quantiles(fib19, sources)
    malecns_quantiles = _quantiles(malecns, sources)
    outside = {
        source: float(
            np.mean(
                [
                    row["four_source_ratios"][source] < fib19_quantiles[source][0]
                    or row["four_source_ratios"][source] > fib19_quantiles[source][-1]
                    for row in malecns
                ]
            )
        )
        for source in sources
    }
    if not all(
        np.isclose(outside[source], value, rtol=0.0, atol=1e-15)
        for source, value in expected["malecns_fraction_outside_fib19_range"].items()
    ):
        raise ValueError("MaleCNS outside-FIB19-range fraction changed")

    fib19_by_subtype = {
        subtype: _mean_ratios(
            [row for row in fib19 if row["target_subtype"] == subtype], sources
        )
        for subtype in expected["target_subtypes"]
    }
    malecns_by_subtype = {
        subtype: _mean_ratios(
            [row for row in malecns if row["target_subtype"] == subtype], sources
        )
        for subtype in expected["target_subtypes"]
    }
    malecns_by_side = {
        side: _mean_ratios([row for row in malecns if row["side"] == side], sources)
        for side in ("L", "R")
    }
    gates = {
        "FIB19_twenty_T5_four_source_ratios_verified": True,
        "MaleCNS_T5_four_source_ratios_verified": True,
        "population_source_rank_order_matches": sorted(sources, key=fib19_mean.get)
        == sorted(sources, key=malecns_mean.get),
        "population_four_source_ratios_exactly_match": np.allclose(
            list(fib19_mean.values()), list(malecns_mean.values()), rtol=0.0, atol=0.0
        ),
        "every_MaleCNS_target_within_FIB19_per_source_ranges": all(
            value == 0.0 for value in outside.values()
        ),
        "FIB19_to_MaleCNS_target_identity_crosswalk_available": False,
        "five_source_weight_mapping_complete": mapping[
            "T5_all_five_source_mapping_complete"
        ],
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(model_path): _sha256(root / model_path),
                str(target_path): _sha256(root / target_path),
                str(mapping_path): _sha256(root / mapping_path),
                config["fib19_source_table"]["path"]: _sha256(source_table),
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "FIB19": {
            "T5_count": len(fib19),
            "subtype_counts": subtype_counts,
            "population_mean_four_source_ratios": fib19_mean,
            "four_source_ratio_quantiles_min_q25_median_q75_max": fib19_quantiles,
            "mean_ratios_by_subtype": fib19_by_subtype,
        },
        "MaleCNS": {
            "T5_count": len(target["T5"]["targets"]),
            "four_source_nonzero_count": len(malecns),
            "zero_four_source_body_ids": zero_four_source,
            "population_mean_four_source_ratios": malecns_mean,
            "four_source_ratio_quantiles_min_q25_median_q75_max": malecns_quantiles,
            "mean_ratios_by_subtype": malecns_by_subtype,
            "mean_ratios_by_side": malecns_by_side,
            "fraction_outside_FIB19_per_source_range": outside,
        },
        "comparison": {
            "population_ratio_difference_MaleCNS_minus_FIB19": {
                source: float(differences[index]) for index, source in enumerate(sources)
            },
            "mean_absolute_population_ratio_difference": float(
                np.mean(np.abs(differences))
            ),
            "maximum_population_ratio_difference": float(np.max(np.abs(differences))),
            "FIB19_mean_rank_descending": sorted(
                sources, key=fib19_mean.get, reverse=True
            ),
            "MaleCNS_mean_rank_descending": sorted(
                sources, key=malecns_mean.get, reverse=True
            ),
        },
        "gates": gates,
        "authorize_FIB19_weight_transfer_to_MaleCNS": False,
        "authorize_Tm_to_T5_model_transfer_to_v7": False,
        "authorize_T5_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "stop_reason": (
            "source_rank_matches_but_ratios_and_target_distributions_do_not_and_no_identity_crosswalk_exists"
        ),
        "cross_check": {
            "author_model_transfer_authorized": model[
                "authorize_Tm_to_T5_model_transfer_to_v7"
            ]
        },
        "boundary": config["boundary"],
    }
