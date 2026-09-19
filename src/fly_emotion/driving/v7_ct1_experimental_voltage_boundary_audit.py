"""Bound claims about experimental CT1 voltage across audited candidates."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from xml.etree import ElementTree

import yaml
from pypdf import PdfReader

from fly_emotion.driving.v7_geometry_sign import _sha256

CONFIG = Path("configs/driving-v7-ct1-experimental-voltage-boundary-audit.yaml")
IMPLEMENTATION = Path(
    "src/fly_emotion/driving/v7_ct1_experimental_voltage_boundary_audit.py"
)


def _verify_file(root: Path, spec: dict) -> Path:
    path = root / spec["path"]
    if path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"CT1 candidate size mismatch: {path}")
    if _sha256(path) != spec["sha256"]:
        raise ValueError(f"CT1 candidate SHA-256 mismatch: {path}")
    return path


def _xml_text(path: Path) -> str:
    return " ".join("".join(ElementTree.parse(path).getroot().itertext()).split())


def evaluate_v7_ct1_experimental_voltage_boundary_audit(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    contract_path = Path(config["required_contract"])
    contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    evidence_paths = {name: Path(path) for name, path in config["evidence"].items()}
    evidence = {
        name: json.loads((root / path).read_text(encoding="utf-8"))
        for name, path in evidence_paths.items()
    }
    documents = {}
    document_text = {}
    for name, spec in config["candidate_documents"].items():
        path = _verify_file(root, spec)
        if spec["format"] == "pdf":
            reader = PdfReader(path, strict=False)
            if len(reader.pages) != int(spec["pages"]):
                raise ValueError(f"CT1 candidate page count changed: {name}")
            text = " ".join(
                " ".join((page.extract_text() or "").split()) for page in reader.pages
            )
            attachments = sorted(reader.attachments or {})
            documents[name] = {
                **spec,
                "actual_sha256": _sha256(path),
                "actual_pages": len(reader.pages),
                "embedded_attachment_names": attachments,
            }
        else:
            text = _xml_text(path)
            documents[name] = {**spec, "actual_sha256": _sha256(path)}
        document_text[name] = text
    search_snapshots = {}
    for name, spec in config["incremental_search_snapshots"].items():
        path = _verify_file(root, spec)
        if name == "Cornean_code_archive":
            search_snapshots[name] = {"verified_archive_path": str(path)}
            continue
        search_snapshots[name] = json.loads(path.read_text(encoding="utf-8"))

    ramos = document_text["Ramos_Traslosheros_2021"]
    if any(
        phrase not in ramos
        for phrase in (
            "axon terminals in the lobula layer 1 of CT1 neurons expressing GCaMP6f",
            "in vivo two-photon calcium imaging",
        )
    ):
        raise ValueError("Ramos CT1 Lo1 measurement text changed")

    groschner = document_text["Groschner_2022"]
    required_groschner = (
        "CT1 > GC6f",
        "Tm3 > ArcLD",
        "Using Arclight to measure membrane voltage",
        "made their calcium responses significantly faster",
        "terminals in both the medulla",
        "and the lobula",
        "All original modeling code has been deposited at GitHub",
    )
    if any(phrase not in groschner for phrase in required_groschner):
        raise ValueError("Groschner CT1 measurement or availability text changed")
    if "CT1 > ArcLD" in groschner or "CT1 > ArcLight" in groschner:
        raise ValueError("Groschner now appears to contain a CT1 ArcLight genotype")
    if "whole-cell" in groschner.lower() or "patch-clamp" in groschner.lower():
        raise ValueError("Groschner now appears to contain CT1 electrophysiology")

    braun_article = document_text["Braun_2023_article"]
    if any(
        phrase not in braun_article
        for phrase in (
            "Using two-photon calcium imaging in combination with thermogenetics",
            "Data and code for analysis and modeling are publicly available",
            "https://doi.org/10.17617/3.QE3MFT",
        )
    ):
        raise ValueError("Braun CT1 functional evidence text changed")
    if any(term in braun_article for term in ("ArcLight", "whole-cell", "patch-clamp")):
        raise ValueError("Braun candidate now appears to contain direct voltage recording")

    samara = document_text["Samara_Borst_2025"]
    if any(
        phrase not in samara
        for phrase in (
            "we use the FlyWire database",
            "Tm and CT1 cells wire on T5a dendrites via eight polyadic synapse types",
            "The data are available at Zenodo",
        )
    ):
        raise ValueError("Samara-Borst CT1 structural evidence changed")
    if any(term in samara.lower() for term in ("whole-cell", "patch clamp", "voltage imaging")):
        raise ValueError("Samara-Borst now appears to contain direct voltage recording")

    henning = document_text["Henning_2026"]
    if any(
        phrase not in henning
        for phrase in (
            "using in vivo two-photon calcium imaging",
            "In vivo calcium imaging data produced for this study",
            "large amacrine cell CT1",
        )
    ):
        raise ValueError("Henning incremental CT1 context changed")
    if any(term in henning.lower() for term in ("whole-cell", "patch clamp", "voltage imaging")):
        raise ValueError("Henning now appears to contain direct voltage recording")

    okuno = document_text["Okuno_2026"]
    if any(
        phrase not in okuno
        for phrase in (
            "whole-brain calcium imaging data",
            "CT1-R neuron in the FlyWire",
            "Whole-brain calcium imaging data and walking behavior data",
        )
    ):
        raise ValueError("Okuno CT1 structural/calcium evidence changed")
    if any(term in okuno.lower() for term in ("whole-cell", "patch clamp", "voltage imaging")):
        raise ValueError("Okuno now appears to contain direct voltage recording")

    cornean = document_text["Cornean_2024"]
    if any(
        phrase not in cornean
        for phrase in (
            "Tm9 neurons expressing GCaMP6f upon optogenetic activation",
            "CT1, synapses onto Tm9 axon terminals in each column",
            "Source data of this study can be found on Zenodo",
        )
    ):
        raise ValueError("Cornean CT1 structural or Tm9 calcium evidence changed")
    if any(term in cornean.lower() for term in ("whole-cell", "patch clamp", "voltage imaging")):
        raise ValueError("Cornean now appears to contain direct voltage recording")

    title_results = search_snapshots["CT1_title_abstract"]
    if title_results["hitCount"] != 4:
        raise ValueError("incremental CT1 title/abstract result count changed")
    result_dois = sorted(
        item.get("doi") for item in title_results["resultList"]["result"]
    )
    expected_dois = sorted(
        [
            "10.1021/acsmedchemlett.6c00155",
            "10.1101/2025.08.04.668437",
            "10.1186/s40246-026-00913-2",
            "10.1371/journal.pone.0334925",
        ]
    )
    if result_dois != expected_dois:
        raise ValueError("incremental CT1 title/abstract candidates changed")
    if search_snapshots["complex_tangential"]["hitCount"] != 0:
        raise ValueError("incremental complex-tangential result count changed")
    okuno_crossref = search_snapshots["Okuno_Crossref"]["message"]
    if okuno_crossref["DOI"].lower() != "10.7554/elife.107990.2":
        raise ValueError("Okuno Crossref identity changed")
    all_time = search_snapshots["CT1_neuron_all_time"]
    if all_time["hitCount"] != 5:
        raise ValueError("all-time CT1-neuron result count changed")
    all_time_dois = sorted(
        item.get("doi") for item in all_time["resultList"]["result"]
    )
    expected_all_time_dois = sorted(
        [
            "10.1016/j.cub.2023.05.007",
            "10.1038/s41467-021-24986-w",
            "10.1038/s41467-024-45971-z",
            "10.1101/2023.08.29.555204",
            "10.7554/elife.57443",
        ]
    )
    if all_time_dois != expected_all_time_dois:
        raise ValueError("all-time CT1-neuron candidates changed")
    cornean_zenodo = search_snapshots["Cornean_Zenodo"]
    if (
        cornean_zenodo["doi"] != "10.5281/zenodo.10361475"
        or len(cornean_zenodo["files"]) != 1
        or cornean_zenodo["files"][0]["key"] != "data.zip"
        or int(cornean_zenodo["files"][0]["size"]) != 31_146_518_166
    ):
        raise ValueError("Cornean Zenodo payload manifest changed")
    code_spec = config["incremental_search_snapshots"]["Cornean_code_archive"]
    with tarfile.open(root / code_spec["path"], "r:gz") as archive:
        readme_name = next(
            name
            for name in archive.getnames()
            if name.endswith("2P-analysis/optogenetic_activation/README.md")
        )
        cornean_code_text = archive.extractfile(readme_name).read().decode("utf-8")
    if any(
        phrase not in cornean_code_text
        for phrase in (
            "Tm9GCaMP6f-L3CsChrimson",
            "170928_lr_L3_C3_Tm1_OptogeneticsWoNaNsWoBGSubstraction.mat",
        )
    ):
        raise ValueError("Cornean code archive target-cell semantics changed")

    ramos_report = evidence["ramos_source_data"]
    meier_report = evidence["meier_extreme_compartmentalization"]
    groschner_report = evidence["groschner_compartment"]
    yang_report = evidence["yang_voltage"]
    kohn_report = evidence["kohn_portes_voltage"]
    flyvis_report = evidence["flyvis_model"]
    if ramos_report["verified_spatial_blocks"]["CT1_OFF_horizontal"]["indicator"] != "GCaMP6f":
        raise ValueError("Ramos CT1 indicator changed")
    if not meier_report["model_evidence"]["output_is_simulated"]:
        raise ValueError("Meier model-voltage boundary changed")
    if groschner_report["T5_lobula_CT1_numerical_payload_verified"]:
        raise ValueError("Groschner Lo1 numerical payload status changed")
    if "CT1" not in yang_report["T5_source_contract"]["sources_without_optical_voltage_phenotype"]:
        raise ValueError("Yang CT1 voltage coverage changed")
    if kohn_report["T5_source_contract"]["missing_numeric_membrane_voltage_sources"] != ["CT1"]:
        raise ValueError("Kohn-Portes CT1 voltage coverage changed")
    if "CT1(Lo1)" not in flyvis_report["source_time_constants"]:
        raise ValueError("FlyVis CT1(Lo1) model source changed")

    candidates = config["candidates"]
    direct_ct1_voltage_candidates = [
        name
        for name, item in candidates.items()
        if item["direct_CT1_measurement"]
        and item["direct_CT1_experimental_membrane_voltage"]
    ]
    lo1_direct_voltage_candidates = [
        name
        for name, item in candidates.items()
        if "lobula_Lo1" in item["compartments"]
        and item["direct_CT1_measurement"]
        and item["direct_CT1_experimental_membrane_voltage"]
    ]
    allowed_units = set(contract["allowed_response_units"])
    gates = {
        "bounded_candidate_documents_verified": True,
        "direct_CT1_experimental_voltage_phenotype_found": bool(
            direct_ct1_voltage_candidates
        ),
        "direct_CT1_Lo1_experimental_voltage_found": bool(lo1_direct_voltage_candidates),
        "direct_CT1_public_numeric_voltage_payload_found": False,
        "allowed_response_unit_available": bool(direct_ct1_voltage_candidates)
        and "millivolts" in allowed_units,
        "stable_biological_individual_ids_available": any(
            item["direct_CT1_experimental_membrane_voltage"]
            and item["stable_biological_individual_ids"]
            for item in candidates.values()
        ),
        "source_to_MaleCNS_mapping_available": False,
        "required_split_roles_available": False,
        "external_final_commitment_available": False,
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
                **{str(path): _sha256(root / path) for path in evidence_paths.values()},
                **{
                    spec["path"]: _sha256(root / spec["path"])
                    for spec in config["incremental_search_snapshots"].values()
                },
            },
            "parameter_fit": False,
            "runtime_modified": False,
        },
        "candidate_documents": documents,
        "candidate_matrix": candidates,
        "candidate_summary": {
            "audited_candidate_count": len(candidates),
            "direct_CT1_measurement_candidates": sorted(
                name for name, item in candidates.items() if item["direct_CT1_measurement"]
            ),
            "direct_CT1_experimental_voltage_candidates": (
                direct_ct1_voltage_candidates
            ),
            "direct_CT1_Lo1_experimental_voltage_candidates": (
                lo1_direct_voltage_candidates
            ),
            "claim_scope": "bounded_audited_candidate_set_not_global_nonexistence",
        },
        "all_time_index_search": {
            "query": '"CT1 neuron" AND Drosophila',
            "Europe_PMC_hit_count": all_time["hitCount"],
            "new_relevant_candidate": "Cornean_2024",
            "structural_context_only_DOIs": ["10.7554/elife.57443"],
            "Cornean_public_dataset_DOI": cornean_zenodo["doi"],
            "Cornean_public_dataset_archive_bytes": int(
                cornean_zenodo["files"][0]["size"]
            ),
            "Cornean_functional_target_cell": "Tm9",
            "new_direct_CT1_experimental_voltage_candidates": [],
        },
        "incremental_search_2025_2026": {
            "date_window": config["search_scope"]["incremental_window"],
            "Europe_PMC_CT1_title_abstract_hit_count": title_results["hitCount"],
            "Europe_PMC_complex_tangential_hit_count": search_snapshots[
                "complex_tangential"
            ]["hitCount"],
            "relevant_candidates": [
                "Samara_Borst_2025",
                "Henning_2026",
                "Okuno_2026",
            ],
            "same_token_false_positive_DOIs": [
                "10.1021/acsmedchemlett.6c00155",
                "10.1186/s40246-026-00913-2",
            ],
            "new_direct_CT1_experimental_voltage_candidates": [],
        },
        "transfer_gates": gates,
        "CT1_experimental_voltage_transfer_authorized": transferable,
        "authorize_T5_functional_precheck": transferable,
        "advance_to_LPLC_mechanism_repair": False,
        "advance_to_vehicle_experiments": False,
        "blocking_requirements": [name for name, passed in gates.items() if not passed],
        "stop_reason": (
            None
            if transferable
            else "audited_CT1_evidence_is_calcium_functional_or_simulated_not_experimental_voltage"
        ),
        "search_scope": config["search_scope"],
        "boundary": config["boundary"],
    }
