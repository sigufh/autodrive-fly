"""Audit independent Mi1 whole-cell voltage and public payload availability."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path(
    "configs/driving-v7-matulis-mi1-voltage-availability-audit.yaml"
)
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_matulis_mi1_voltage_availability_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"Matulis Mi1 source size changed: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"Matulis Mi1 source SHA-256 changed: {path}")
    return path


def _xml_text(path: Path) -> str:
    return " ".join(
        "".join(ElementTree.parse(path).getroot().itertext()).split()
    )


def evaluate_v7_matulis_mi1_voltage_availability_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))

    paper_path = _verify_file(root, config["paper"])
    paper_text = _xml_text(paper_path)
    required_paper_phrases = (
        "patch clamped individual Mi1 neurons and recorded membrane potential",
        "Electrophysiology data from Mi1 was acquired at 10 kHz",
        "Membrane voltage was recorded paired with timing information",
        "120 Hz flicker was used for electrophysiology recordings",
        "N = 3 cells in 3 flies",
        "have not been deposited in a public repository",
        "available from the corresponding author on request",
    )
    if any(phrase not in paper_text for phrase in required_paper_phrases):
        raise ValueError("Matulis Mi1 voltage or availability text changed")

    biostudies_path = _verify_file(root, config["biostudies"])
    biostudies = json.loads(biostudies_path.read_text(encoding="utf-8"))
    if biostudies["accno"] != config["biostudies"]["accession"]:
        raise ValueError("Matulis BioStudies accession changed")
    biostudies_files = {
        item["path"]: int(item["size"])
        for item in biostudies["section"]["files"]
    }
    if biostudies_files != config["biostudies"]["expected_files"]:
        raise ValueError("Matulis BioStudies attachment list changed")

    docx_spec = config["supplements"]["key_resources"]
    docx_path = _verify_file(root, docx_spec)
    with zipfile.ZipFile(docx_path) as archive:
        members = sorted(name for name in archive.namelist() if not name.endswith("/"))
        embedded = sorted(
            name for name in members if name.startswith("word/embeddings/")
        )
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        document_text = " ".join(
            "".join(document.itertext()).split()
        )
        embedded_payload = archive.read(embedded[0])
    if len(members) != int(docx_spec["expected_member_count"]):
        raise ValueError("Matulis key-resources DOCX member count changed")
    if embedded != sorted(docx_spec["expected_embedded_objects"]):
        raise ValueError("Matulis key-resources embedded-object set changed")
    if b"Adobe Photoshop Image" not in embedded_payload or b"8BPS" not in embedded_payload:
        raise ValueError("Matulis embedded object no longer identifies as an image")
    if any(
        phrase not in document_text
        for phrase in (
            "Custom stimulus presentation and data acquisition code",
            "Custom analysis code",
            "Available upon request",
        )
    ):
        raise ValueError("Matulis key-resources availability text changed")

    pdf_spec = config["supplements"]["figures"]
    pdf_path = _verify_file(root, pdf_spec)
    pdf = PdfReader(pdf_path, strict=False)
    if len(pdf.pages) != int(pdf_spec["pages"]):
        raise ValueError("Matulis supplement page count changed")
    if sorted(pdf.attachments or {}):
        raise ValueError("Matulis supplement gained an unreviewed attachment")
    pdf_text = " ".join(
        " ".join((page.extract_text() or "").split()) for page in pdf.pages
    )
    if "Mi1 membrane voltage for a single fly" not in pdf_text:
        raise ValueError("Matulis Mi1 supplement caption changed")

    measurement = config["measurement"]
    allowed_unit = measurement["response_unit"] in set(
        contract["allowed_response_units"]
    )
    gates = {
        "independent_Mi1_experimental_voltage_phenotype_verified": True,
        "allowed_response_unit_published": allowed_unit,
        "one_cell_per_reported_fly_count": (
            measurement["reported_cell_count"]
            == measurement["reported_fly_count"]
        ),
        "public_numeric_voltage_payload_available": False,
        "stable_pseudonymous_biological_individual_ids_available": False,
        "complete_record_specific_stimulus_fields_available": False,
        "training_validation_external_final_roles_available": False,
    }
    return {
        "protocol": {
            "name": config["name"],
            "observed_on": config["observed_on"],
            "dependencies_sha256": {
                str(CONFIG): _sha256(root / CONFIG),
                str(IMPLEMENTATION): _sha256(root / IMPLEMENTATION),
                str(contract_path): _sha256(root / contract_path),
                config["paper"]["path"]: _sha256(paper_path),
                config["biostudies"]["path"]: _sha256(biostudies_path),
                docx_spec["path"]: _sha256(docx_path),
                pdf_spec["path"]: _sha256(pdf_path),
            },
            "paper": config["paper"],
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "measurement": measurement,
        "public_attachment_inventory": {
            "BioStudies_accession": biostudies["accno"],
            "files": biostudies_files,
            "numeric_data_file_count": 0,
            "key_resources_DOCX_member_count": len(members),
            "embedded_object_names": embedded,
            "embedded_object_classification": "Adobe_Photoshop_image",
            "supplement_PDF_page_count": len(pdf.pages),
            "supplement_PDF_attachment_names": sorted(pdf.attachments or {}),
        },
        "data_and_code_availability": {
            "public_repository_deposit_declared": False,
            "available_upon_request_only": True,
            "local_numeric_voltage_payload_verified": False,
            "global_absence_claimed": False,
        },
        "transfer_gates": gates,
        "independent_Mi1_voltage_transfer_authorized": False,
        "authorize_T4_functional_precheck": False,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [
            name for name, passed in gates.items() if not passed
        ],
        "stop_reason": "independent_Mi1_voltage_is_published_but_numeric_payload_is_upon_request",
        "boundary": config["boundary"],
    }
