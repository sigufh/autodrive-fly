import hashlib
import json
from pathlib import Path

import numpy as np

from fly_emotion.driving.v7_stage1_input_audit import (
    encode_trace,
    receptor_separability_control,
)

ROOT = Path(__file__).parents[1]


def test_retinal_encodings_preserve_receptor_separability_and_signed_changes() -> None:
    values = np.array([[0.2, 0.8], [0.4, 0.6], [0.1, 0.9]], dtype=np.float32)
    linear = encode_trace(values, "linear_luminance")
    np.testing.assert_array_equal(linear, values)
    difference = encode_trace(values, "signed_frame_difference")
    np.testing.assert_array_equal(difference[0], [0, 0])
    assert difference[1, 0] > 0 and difference[1, 1] < 0
    for encoding in ("linear_luminance", "signed_frame_difference"):
        control = receptor_separability_control(encoding)
        assert control["maximum_other_receptor_change"] == 0.0
        assert control["maximum_target_receptor_change"] > 1e-6
        assert control["encoding_reads_direction_label"] is False


def test_saved_development_input_audit_passes_without_neural_evaluation() -> None:
    report = json.loads((ROOT / "artifacts/v7-stage1-input-audit.json").read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["split"] == "development"
    assert report["protocol"]["neural_evaluation_performed"] is False
    assert report["stimulus_count"] == 172
    assert report["input_gates_pass"] is True
    assert report["development_neural_evaluation_allowed"] is True
    assert report["advance_to_validation"] is False
    assert report["advance_to_central_complex"] is False
    assert report["retinal_maps"]["default_3344"]["receptor_count"] == 3344
    assert report["retinal_maps"]["nested_t4_axis_3344"]["receptor_count"] == 3344
    assert report["retinal_maps"]["balanced_1914"]["receptor_count"] == 1914
    for mapping in report["retinal_maps"].values():
        for result in mapping["encodings"].values():
            assert result["passed"] is True
            assert result["gates"]["all_on_off_pairs_have_opposite_sign"] is True
            assert result["gates"]["retinal_encoding_is_receptor_separable"] is True
    balanced = report["retinal_maps"]["balanced_1914"]["encodings"]
    assert all(
        item["mirror_summary"]["maximum_absolute_error"] == 0.0 for item in balanced.values()
    )
