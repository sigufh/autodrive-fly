import hashlib
from pathlib import Path

from fly_emotion.driving.v7_mirror_audit import (
    MIRROR_AUDIT_IMPLEMENTATION,
    LayerMirrorProbe,
    evaluate_v7_layerwise_mirror_audit,
)

ROOT = Path(__file__).parents[1]


def test_v7_layer_mirror_probe_includes_exactly_balanced_receptors_and_visual_layers() -> None:
    probe = LayerMirrorProbe(ROOT)
    assert len(probe.layer_populations["R1-R6_L"]) == 957
    assert len(probe.layer_populations["R1-R6_R"]) == 957
    for cell_type in ("L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "C3"):
        assert f"{cell_type}_L" in probe.layer_populations
        assert f"{cell_type}_R" in probe.layer_populations


def test_v7_layerwise_mirror_audit_starts_at_exact_receptor_input_and_cannot_advance() -> None:
    report = evaluate_v7_layerwise_mirror_audit(ROOT)
    assert (
        report["protocol"]["implementation_sha256"]
        == hashlib.sha256((ROOT / MIRROR_AUDIT_IMPLEMENTATION).read_bytes()).hexdigest()
    )
    assert (
        report["protocol"]["v7_implementation_sha256"]
        == hashlib.sha256((ROOT / "src/fly_emotion/driving/v7.py").read_bytes()).hexdigest()
    )
    receptor = report["layer_summary"]["R1-R6"]
    assert receptor["maximum_mirror_error"] == 0.0
    assert report["first_layer_above_0_20_mean_error"] != "R1-R6"
    assert report["active_response_scale_threshold"] == 1e-3
    assert report["first_active_layer_above_0_20_mean_error"] != "R1-R6"
    assert report["advance_to_central_complex"] is False
