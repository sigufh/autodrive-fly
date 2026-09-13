from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from fly_emotion.connectome.overview import build_overview


def test_overview_uses_real_positions_and_keeps_canonical_count(tmp_path: Path) -> None:
    source = tmp_path / "annotations.feather"
    feather.write_feather(
        pa.table(
            {
                "bodyId": [1, 2, 3],
                "status": ["Traced", "Traced", "Orphan"],
                "superclass": ["cb_sensory", "vnc_motor", None],
                "somaLocation": [[1000, 2000, 3000], None, [9, 9, 9]],
                "tosomaLocation": [None, [4000, 5000, 6000], None],
            }
        ),
        source,
    )
    result = build_overview(source, tmp_path / "overview.json")
    assert result["canonical_nodes"] == 2
    assert result["positioned_nodes"] == 2
    assert result["positions"][0] == [8.0, 16.0, 24.0]
