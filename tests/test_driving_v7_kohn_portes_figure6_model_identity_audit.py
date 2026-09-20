import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-kohn-portes-figure6-model-identity-audit.json"


def test_figure6_model_identity_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_official_widget_manifests_bind_both_archives() -> None:
    report = json.loads(REPORT.read_text())
    model = report["official_manifests"]["model"]
    supporting = report["official_manifests"]["supporting"]
    assert (model["id"], model["size"], model["md5"]) == (
        31067761, 3_630_019, "14ba0fa761a513d55cacc41610881e80"
    )
    assert (supporting["id"], supporting["size"], supporting["md5"]) == (
        30862813, 17_374, "3d899bec066f152062015d159f46bc6e"
    )
    assert report["gates"]["official_widget_manifests_match_recovered_packages"]


def test_matlab_target_model_does_not_substitute_for_figure6_axolotl() -> None:
    report = json.loads(REPORT.read_text())
    identity = report["model_identity"]
    assert identity["official_model_entrypoint"] == "t5_simple_wrap"
    assert identity["official_model_components"] == ["E", "I", "E2", "I2"]
    assert set(identity["official_model_source_type_mentions"].values()) == {False}
    assert identity["Figure6_import"] == "from axolotl.tmodel import models, stimuli"
    assert identity["same_model_implementation"] is False
    gates = report["gates"]
    assert gates["official_MATLAB_T5_target_model_verified"] is True
    assert gates["Figure6_axolotl_import_verified"] is True
    assert gates["Figure6_axolotl_local_path_verified"] is True
    assert gates["Figure6_axolotl_source_recovered"] is False
    assert report["T5_source_mapping_available"] is False
    assert report["Figure6_model_reproducible_from_recovered_packages"] is False
    assert report["authorize_Tm_to_T5_model_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
