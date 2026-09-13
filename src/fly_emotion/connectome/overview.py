from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.feather as feather


def build_overview(annotations_path: Path, output_path: Path) -> dict:
    table = feather.read_table(
        annotations_path,
        columns=["bodyId", "status", "superclass", "somaLocation", "tosomaLocation"],
        memory_map=True,
    )
    table = table.filter(pc.is_valid(table["superclass"]))
    classes = sorted({str(value) for value in table["superclass"].to_pylist() if value})
    class_index = {name: index for index, name in enumerate(classes)}
    body_ids: list[int] = []
    positions: list[list[float]] = []
    class_ids: list[int] = []
    for row in table.to_pylist():
        position = row["somaLocation"] or row["tosomaLocation"]
        if not position:
            continue
        body_ids.append(int(row["bodyId"]))
        # Annotation coordinates use 8 nm voxels. Convert them to micrometres.
        positions.append([float(value) * 0.008 for value in position])
        class_ids.append(class_index.get(str(row["superclass"]), -1))
    payload = {
        "release": "male-cns:v1.0",
        "canonical_nodes": table.num_rows,
        "positioned_nodes": len(body_ids),
        "units": "micrometres",
        "body_ids": body_ids,
        "positions": positions,
        "class_ids": class_ids,
        "classes": classes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload
