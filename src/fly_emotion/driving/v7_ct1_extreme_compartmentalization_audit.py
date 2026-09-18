"""Separate measured CT1 calcium from simulated CT1 membrane voltage."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import yaml

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-ct1-extreme-compartmentalization-audit.yaml")
IMPLEMENTATION = Path("src/fly_emotion/driving/v7_ct1_extreme_compartmentalization_audit.py")


def evaluate_v7_ct1_extreme_compartmentalization_audit(root: Path) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("the CT1 audit requires the pypdf development dependency") from exc

    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    supplements = {}
    combined_text = []
    for name, spec in config["official_supplements"].items():
        path = root / spec["path"]
        if path.stat().st_size != int(spec["bytes"]) or _sha256(path) != spec["sha256"]:
            raise ValueError(f"official CT1 supplement changed: {name}")
        reader = PdfReader(path)
        if len(reader.pages) != int(spec["pages"]):
            raise ValueError(f"official CT1 supplement page count changed: {name}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        combined_text.append(text)
        supplements[name] = {
            **spec,
            "actual_sha256": _sha256(path),
            "actual_pages": len(reader.pages),
            "embedded_attachment_names": sorted(reader.attachments),
        }
    paper_text = "\n".join(combined_text)
    normalized_paper_text = " ".join(paper_text.replace("ﬂ", "fl").split())
    required_phrases = (
        "Using calcium imaging and compartmental modeling",
        "Relative fluorescence changes",
        "raw calcium traces",
        "Python programs used for the compartmental model",
        "10.5281/zenodo.2636606",
    )
    if any(phrase not in normalized_paper_text for phrase in required_phrases):
        raise ValueError("CT1 paper evidence text changed")

    archive_spec = config["zenodo"]["archive"]
    archive = root / archive_spec["path"]
    if archive.stat().st_size != int(archive_spec["bytes"]):
        raise ValueError("CT1 Zenodo archive size mismatch")
    if _sha256(archive) != archive_spec["sha256"]:
        raise ValueError("CT1 Zenodo archive SHA-256 mismatch")
    if (
        hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
        != archive_spec["publisher_md5"]
    ):
        raise ValueError("CT1 Zenodo publisher MD5 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(name for name in zipped.namelist() if not name.endswith("/"))
        if members != sorted(config["zenodo"]["expected_members"]):
            raise ValueError("CT1 Zenodo member list changed")
        model_name = next(name for name in members if name.endswith("CT1CompModeling.py"))
        model_code = zipped.read(model_name).decode("utf-8")
    required_model_fragments = (
        "curramp=10*(10**(-12))",
        "Vm = spsolve(M,currinj)",
        "Vm = 1000.0*Vm # mV",
    )
    if any(fragment not in model_code for fragment in required_model_fragments):
        raise ValueError("CT1 model voltage provenance changed")
    generated_outputs = sorted(set(re.findall(r"np\.save\(['\"]([^'\"]+)", model_code)))

    experiment = config["experimental_contract"]
    gates = {
        "official_supplement_PDFs_verified": True,
        "supplement_PDFs_have_no_embedded_attachments": all(
            not item["embedded_attachment_names"] for item in supplements.values()
        ),
        "Zenodo_model_archive_verified": True,
        "experimental_CT1_calcium_phenotype_verified": True,
        "experimental_CT1_numeric_time_series_publicly_deposited": bool(
            experiment["experimental_numeric_time_series_publicly_deposited"]
        ),
        "experimental_CT1_allowed_response_unit_available": bool(
            experiment["experimental_membrane_voltage_measured"]
            and "millivolts" in contract["allowed_response_units"]
        ),
        "stable_biological_individual_ids_available": bool(
            experiment["stable_biological_individual_ids_publicly_deposited"]
        ),
        "source_to_MaleCNS_mapping_available": bool(
            experiment["source_to_MaleCNS_mapping_present"]
        ),
        "required_split_roles_available": bool(experiment["required_split_roles_present"]),
    }
    transferable = all(gates.values())
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
            },
            "paper": config["paper"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "official_supplements": supplements,
        "zenodo_model_release": {
            **config["zenodo"],
            "actual_archive_sha256": _sha256(archive),
            "actual_members": members,
            "model_generated_output_names": generated_outputs,
        },
        "experimental_evidence": experiment,
        "model_evidence": config["model_contract"],
        "transfer_gates": gates,
        "CT1_experimental_source_dynamics_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "published_CT1_calcium_and_simulated_voltage_do_not_supply_experimental_voltage"
        ),
        "boundary": config["boundary"],
    }
