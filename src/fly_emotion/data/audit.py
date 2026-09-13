from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather

CANONICAL_NODE_RULE = "superclass is not null"


def _counts(values: list[Any]) -> dict[str, int]:
    return dict(
        sorted(Counter("<null>" if value is None else str(value) for value in values).items())
    )


def audit_annotations(path: Path) -> tuple[dict[str, Any], set[int]]:
    table = feather.read_table(path, memory_map=True)
    required = {"bodyId", "status", "statusLabel", "superclass", "type"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"annotation columns missing: {sorted(missing)}")

    canonical_table = table.filter(pc.is_valid(table["superclass"]))
    ids = canonical_table["bodyId"].to_pylist()
    canonical = {int(value) for value in ids}
    if len(canonical) != len(ids):
        raise ValueError("duplicate bodyId in canonical annotation rows")

    report = {
        "source_rows": table.num_rows,
        "unique_body_ids": len(set(table["bodyId"].to_pylist())),
        "canonical_rule": CANONICAL_NODE_RULE,
        "canonical_nodes": len(canonical),
        "strictly_traced_nodes": int(pc.sum(pc.equal(table["status"], "Traced")).as_py()),
        "status_counts": _counts(table["status"].to_pylist()),
        "status_label_counts": _counts(table["statusLabel"].to_pylist()),
        "canonical_status_counts": _counts(canonical_table["status"].to_pylist()),
        "superclass_counts": _counts(canonical_table["superclass"].to_pylist()),
        "typed_canonical_nodes": canonical_table.num_rows - canonical_table["type"].null_count,
    }
    return report, canonical


def audit_neurotransmitters(path: Path, canonical: set[int]) -> dict[str, Any]:
    table = feather.read_table(path, memory_map=True)
    required = {"body", "predicted_nt", "predicted_nt_confidence", "consensus_nt"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"neurotransmitter columns missing: {sorted(missing)}")
    bodies = table["body"].to_pylist()
    canonical_mask = [int(body) in canonical for body in bodies]
    filtered = table.filter(canonical_mask)
    return {
        "source_rows": table.num_rows,
        "canonical_rows": filtered.num_rows,
        "canonical_coverage": filtered.num_rows / len(canonical) if canonical else 0.0,
        "predicted_nt_counts": _counts(filtered["predicted_nt"].to_pylist()),
        "consensus_nt_counts": _counts(filtered["consensus_nt"].to_pylist()),
    }


def audit_connections(
    path: Path, canonical: set[int], *, batch_size: int = 4_000_000
) -> dict[str, Any]:
    """Stream the full flat edge table and audit the canonical neuron subgraph."""
    table = feather.read_table(path, memory_map=True, columns=["body_pre", "body_post", "weight"])
    required = {"body_pre", "body_post", "weight"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"connection columns missing: {sorted(missing)}")

    canonical_ids = np.fromiter(canonical, dtype=np.int64, count=len(canonical))
    source_rows = table.num_rows
    canonical_edges = 0
    canonical_synapses = 0
    self_edges = 0
    touched: set[int] = set()
    min_weight: int | None = None
    max_weight = 0
    for batch in table.to_batches(max_chunksize=batch_size):
        pre = batch.column(0).to_numpy(zero_copy_only=False)
        post = batch.column(1).to_numpy(zero_copy_only=False)
        weight = batch.column(2).to_numpy(zero_copy_only=False)
        mask = np.isin(pre, canonical_ids) & np.isin(post, canonical_ids)
        if not mask.any():
            continue
        selected_pre = pre[mask]
        selected_post = post[mask]
        selected_weight = weight[mask]
        canonical_edges += int(mask.sum())
        canonical_synapses += int(selected_weight.sum(dtype=np.int64))
        self_edges += int((selected_pre == selected_post).sum())
        touched.update(map(int, selected_pre))
        touched.update(map(int, selected_post))
        current_min = int(selected_weight.min())
        min_weight = current_min if min_weight is None else min(min_weight, current_min)
        max_weight = max(max_weight, int(selected_weight.max()))

    return {
        "source_rows": source_rows,
        "canonical_edges": canonical_edges,
        "canonical_synapse_weight_sum": canonical_synapses,
        "canonical_nodes_with_edges": len(touched),
        "canonical_isolated_nodes": len(canonical - touched),
        "self_edges": self_edges,
        "minimum_weight": min_weight,
        "maximum_weight": max_weight,
        "filter": "both body_pre and body_post have a non-null neuron superclass",
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
