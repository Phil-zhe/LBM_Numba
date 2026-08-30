from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


SCHEMA_VERSION = 1


def configuration_hash(config_dict):
    encoded = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(path, solver, config_dict, backend, algorithm_state=None):
    current_step = solver.current_step if hasattr(solver, "current_step") else solver.state.current_step
    np.savez_compressed(
        Path(path), checkpoint_schema_version=np.array(SCHEMA_VERSION),
        current_f=solver.copy_current_f(), current_step=np.array(current_step),
        backend=np.array(backend), precision=np.array(str(solver.copy_current_f().dtype)),
        configuration_hash=np.array(configuration_hash(config_dict)),
        macroscopic_valid=np.array(False),
        algorithm_state=np.array(json.dumps(algorithm_state or {}, sort_keys=True)),
    )


def load_checkpoint(path, solver, config_dict, backend):
    with np.load(Path(path), allow_pickle=False) as data:
        if int(data["checkpoint_schema_version"]) != SCHEMA_VERSION:
            raise ValueError("incompatible checkpoint schema")
        if str(data["backend"]) != backend:
            raise ValueError("checkpoint backend mismatch")
        if str(data["configuration_hash"]) != configuration_hash(config_dict):
            raise ValueError("checkpoint configuration mismatch")
        f = data["current_f"]
        expected_dtype = solver.host.f.dtype if hasattr(solver, "host") else solver.state.f.dtype
        if f.dtype != expected_dtype:
            raise ValueError("checkpoint dtype mismatch")
        solver.restore_f(f)
        step = int(data["current_step"])
        if hasattr(solver, "current_step"):
            solver.current_step = step
            solver.macroscopic_valid = False
        else:
            solver.state.current_step = step
            solver.state.macroscopic_valid = False
        return json.loads(str(data["algorithm_state"]))

