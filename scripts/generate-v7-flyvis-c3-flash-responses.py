#!/usr/bin/env python3
"""Generate C3 flash traces from the frozen FlyVis pretrained ensemble.

This script intentionally lives outside the package runtime.  It requires the
FlyVis audit environment and writes only an external-model evidence payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol(name: str, dt: float) -> dict:
    if name == "henning_matched":
        return {
            "name": name,
            "dt_seconds": dt,
            "values": [0.0, 1.0, 0.0],
            "durations_seconds": [5.0, 5.0, 3.0],
            "steady_state_seconds": 0.0,
            "steady_state_value": None,
        }
    if name == "flyvis_official":
        return {
            "name": name,
            "dt_seconds": dt,
            "values": [0.5, 1.0, 0.5],
            "durations_seconds": [1.0, 1.0, 1.0],
            "steady_state_seconds": 1.0,
            "steady_state_value": 0.5,
        }
    raise ValueError(f"unknown protocol: {name}")


def _stimulus_values(protocol: dict) -> np.ndarray:
    segments = [
        np.full(round(duration / protocol["dt_seconds"]), value, dtype=np.float32)
        for value, duration in zip(
            protocol["values"], protocol["durations_seconds"], strict=True
        )
    ]
    return np.concatenate(segments)


def _central_c3_index(network) -> int:
    cell_types = network.connectome.nodes.type[:].astype(str)
    u = network.connectome.nodes.u[:]
    v = network.connectome.nodes.v[:]
    index = np.flatnonzero((cell_types == "C3") & (u == 0) & (v == 0))
    if len(index) != 1:
        raise ValueError(f"expected one central C3 neuron, found {len(index)}")
    return int(index[0])


def _input_contract(network) -> dict:
    cell_types = network.connectome.nodes.type[:].astype(str)
    native_index = np.asarray(network.stimulus.input_index)
    native_types = sorted(set(cell_types[native_index.ravel()]))
    permitted_types = [f"R{index}" for index in range(1, 7)]
    permitted_index = np.concatenate(
        [np.asarray(network.stimulus.layer_index[name]) for name in permitted_types]
    )
    if native_types != [f"R{index}" for index in range(1, 9)]:
        raise ValueError(f"unexpected native FlyVis input cell types: {native_types}")
    if sorted(set(cell_types[permitted_index])) != permitted_types:
        raise ValueError("R1-R6-only input index is invalid")
    return {
        "native_input_cell_types": native_types,
        "permitted_input_cell_types": permitted_types,
        "excluded_input_cell_types": ["R7", "R8"],
        "permitted_input_neuron_count": int(len(permitted_index)),
        "R7_R8_external_activity": 0.0,
        "target_recorded": "C3_u0_v0",
        "target_activity_injected": False,
    }


def _simulate_trace(network, protocol: dict, c3_index: int) -> list[float]:
    dt = float(protocol["dt_seconds"])
    values = _stimulus_values(protocol)
    network.clamp()
    params = network._param_api()
    state = network._initial_state(params, 1)
    steady_frames = round(float(protocol["steady_state_seconds"]) / dt)
    if steady_frames:
        network.stimulus.zero(1, steady_frames)
        steady_value = torch.full(
            (1, steady_frames, 1), float(protocol["steady_state_value"])
        )
        for cell_type in [f"R{index}" for index in range(1, 7)]:
            network.stimulus.buffer[
                :, :, network.stimulus.layer_index[cell_type]
            ] += steady_value
        steady_input = network.stimulus()
        with torch.inference_mode():
            for frame in range(steady_frames):
                state = network._next_state(params, state, steady_input[:, frame], dt)
    network.stimulus.zero(1, len(values))
    external = torch.from_numpy(values).view(1, -1, 1)
    for cell_type in [f"R{index}" for index in range(1, 7)]:
        network.stimulus.buffer[:, :, network.stimulus.layer_index[cell_type]] += external
    network.stimulus._nonzero = True
    inputs = network.stimulus()
    response = np.empty(len(values), dtype=np.float32)
    with torch.inference_mode():
        for frame in range(len(values)):
            state = network._next_state(params, state, inputs[:, frame], dt)
            response[frame] = float(state.nodes.activity[0, c3_index])
    if not np.all(np.isfinite(response)):
        raise ValueError("non-finite FlyVis C3 response")
    return [float(value) for value in response]


def generate(root: Path, output: Path) -> dict:
    from flyvis import NetworkView
    from flyvis import __version__ as flyvis_version

    flyvis_root = Path(os.environ["FLYVIS_ROOT_DIR"]).resolve()
    repository_commit = (flyvis_root / ".git/HEAD").read_text().strip()
    model_root = flyvis_root / "results/flow/0000"
    model_dirs = sorted(
        path for path in model_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if len(model_dirs) != 50:
        raise ValueError(f"expected 50 FlyVis models, found {len(model_dirs)}")
    protocols = [
        _protocol(name, dt)
        for name in ("henning_matched", "flyvis_official")
        for dt in (0.005, 0.02)
    ]
    results = []
    network = None
    input_contract = None
    for model_number, model_dir in enumerate(model_dirs):
        view = NetworkView(f"flow/0000/{model_dir.name}")
        network = view.init_network(network=network)
        network.eval()
        if input_contract is None:
            input_contract = _input_contract(network)
        elif _input_contract(network) != input_contract:
            raise ValueError("FlyVis input contract changed across models")
        c3_index = _central_c3_index(network)
        checkpoint = Path(view.get_checkpoint("best"))
        model = {
            "model_number": model_number,
            "model_name": view.name,
            "checkpoint_path": str(checkpoint.relative_to(flyvis_root)),
            "checkpoint_sha256": _sha256(checkpoint),
            "C3_neuron_index": c3_index,
            "responses": [],
        }
        for protocol in protocols:
            model["responses"].append(
                {**protocol, "C3_activity": _simulate_trace(network, protocol, c3_index)}
            )
        results.append(model)
        print(f"completed {model_number + 1:02d}/50 {view.name}", flush=True)
    payload = {
        "schema_version": 1,
        "generator": str(Path(__file__).resolve().relative_to(root)),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "flyvis_root": str(flyvis_root.relative_to(root)),
        "flyvis_repository_commit": repository_commit,
        "environment": {
            "python": platform.python_version(),
            "flyvis": flyvis_version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "input_contract": input_contract,
        "protocols": protocols,
        "models": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/t4-external-models/flyvis-c3-flash-responses.json"),
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    payload = generate(root, args.output.resolve())
    print(f"wrote {args.output} with {len(payload['models'])} models")


if __name__ == "__main__":
    main()
