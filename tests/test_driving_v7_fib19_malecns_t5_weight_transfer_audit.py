import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
REPORT = ROOT / "artifacts/v7-fib19-malecns-t5-weight-transfer-audit.json"


def test_cross_connectome_weight_audit_is_hash_bound_and_read_only() -> None:
    report = json.loads(REPORT.read_text())
    for path, digest in report["protocol"]["dependencies_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["protocol"]["parameter_fit"] is False
    assert report["protocol"]["runtime_modified"] is False


def test_fib19_and_malecns_weight_distributions_are_not_collapsed() -> None:
    report = json.loads(REPORT.read_text())
    assert report["FIB19"]["T5_count"] == 20
    assert report["FIB19"]["subtype_counts"] == {
        "T5a": 5,
        "T5b": 5,
        "T5c": 5,
        "T5d": 5,
    }
    assert report["MaleCNS"]["T5_count"] == 6719
    assert report["MaleCNS"]["four_source_nonzero_count"] == 6718
    assert report["MaleCNS"]["zero_four_source_body_ids"] == [152777]
    assert np.isclose(
        report["comparison"]["maximum_population_ratio_difference"],
        0.126798822451821,
    )
    assert np.isclose(
        report["MaleCNS"]["fraction_outside_FIB19_per_source_range"]["Tm9"],
        0.7307234295921405,
    )
    assert set(report["MaleCNS"]["mean_ratios_by_subtype"]) == {
        "T5a",
        "T5b",
        "T5c",
        "T5d",
    }
    assert set(report["MaleCNS"]["mean_ratios_by_side"]) == {"L", "R"}


def test_rank_agreement_does_not_authorize_weight_transfer() -> None:
    report = json.loads(REPORT.read_text())
    assert report["gates"]["population_source_rank_order_matches"] is True
    assert report["gates"]["population_four_source_ratios_exactly_match"] is False
    assert report["gates"]["every_MaleCNS_target_within_FIB19_per_source_ranges"] is False
    assert report["gates"]["FIB19_to_MaleCNS_target_identity_crosswalk_available"] is False
    assert report["gates"]["five_source_weight_mapping_complete"] is False
    assert report["authorize_FIB19_weight_transfer_to_MaleCNS"] is False
    assert report["authorize_Tm_to_T5_model_transfer_to_v7"] is False
    assert report["authorize_T5_functional_precheck"] is False
    assert report["advance_to_LPLC_mechanism_repair"] is False
