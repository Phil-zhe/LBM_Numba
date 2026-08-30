from __future__ import annotations

import time
from pathlib import Path
import numpy as np

from ..backends.numpy_cpu import NumpyCPUSolver
from ..config import CavityConfig
from ..diagnostics import field_diagnostics, velocity_residual
from ..io import save_cavity_plots, unique_result_dir, write_centerlines, write_json, write_toml_flat
from ..state import initial_cavity_state
from ..validation.cavity_validation import validate_against_ghia


def make_solver(config: CavityConfig, backend: str):
    dtype = np.float32 if config.precision == "float32" else np.float64
    state = initial_cavity_state(config.nx, config.ny, config.lid_velocity, dtype)
    omega = config.nu_tau_omega[2]
    if backend == "numpy-cpu":
        return NumpyCPUSolver(state, omega), state
    if backend == "numba-cuda-mlir":
        from ..backends.numba_cuda_mlir import CudaMLIRSolver
        return CudaMLIRSolver(state, omega), state
    raise ValueError(f"unknown backend: {backend}")


def run_cavity(config: CavityConfig, backend: str, mode: str,
               results_root="results", reference_dir="reference"):
    solver, state = make_solver(config, backend)
    rho, u = solver.get_fields()
    fluid = ~state.solid
    initial_mass = float(rho[fluid].sum(dtype=np.float64))
    previous = u.copy()
    diagnostics = []
    residual_history = []
    consecutive = 0
    converged = False
    first = field_diagnostics(rho, u, state.solid, initial_mass)
    first["step"] = 0
    diagnostics.append(first)
    start = time.perf_counter()
    for step in range(1, config.max_steps + 1):
        solver.step()
        due_check = mode in {"validation", "full", "smoke"} and step % config.check_interval == 0
        due_diag = mode in {"validation", "full", "smoke"} and step % config.diagnostic_interval == 0
        final = step == config.max_steps
        if due_check or due_diag or final:
            rho, u = solver.get_fields()
        if due_diag or final:
            d = field_diagnostics(rho, u, state.solid, initial_mass)
            d["step"] = step
            diagnostics.append(d)
            if d["nonfinite_count"] or d["min_rho"] <= 0:
                raise FloatingPointError(f"unstable field at step {step}: {d}")
        if due_check:
            residual = velocity_residual(u, previous, state.solid)
            residual_history.append({"step": step, "residual": residual})
            previous[...] = u
            consecutive = consecutive + 1 if step >= config.minimum_steps and residual < config.convergence_tolerance else 0
            if consecutive >= config.consecutive_converged_checks:
                converged = True
                break
    if hasattr(solver, "synchronize"):
        solver.synchronize()
    elapsed = time.perf_counter() - start
    rho, u = solver.get_fields()
    actual_step = solver.current_step if hasattr(solver, "current_step") else state.current_step
    if diagnostics[-1]["step"] != actual_step:
        d = field_diagnostics(rho, u, state.solid, initial_mass)
        d["step"] = actual_step
        diagnostics.append(d)
    metrics, lines = validate_against_ghia(rho, u, config.lid_velocity, reference_dir)
    summary = {
        **config.to_dict(), "backend": backend, "mode": mode,
        "steps": actual_step, "converged": converged,
        "termination_reason": "converged" if converged else "max_steps_reached",
        "wall_time_seconds": elapsed,
        "final_residual": residual_history[-1]["residual"] if residual_history else None,
        "diagnostic_steps": [d["step"] for d in diagnostics],
        "rho_deviation_final": diagnostics[-1]["rho_deviation"],
        "rho_deviation_peak": max(d["rho_deviation"] for d in diagnostics),
        "ma_max_final": diagnostics[-1]["ma_max"],
        "ma_max_peak": max(d["ma_max"] for d in diagnostics),
        "mass_drift": diagnostics[-1]["mass_drift"],
        "density_min": diagnostics[-1]["min_rho"],
        "density_max": diagnostics[-1]["max_rho"],
        **metrics,
    }
    result_dir = unique_result_dir(results_root, f"cavity_{backend}")
    np.savez_compressed(result_dir / "fields_final.npz", rho=rho, u_x=u[0], u_y=u[1],
                        solid=state.solid, current_step=np.array(actual_step))
    write_json(result_dir / "diagnostics.json", {"summary": summary, "diagnostics": diagnostics,
                                                  "residual_history": residual_history})
    write_json(result_dir / "run_metadata.json", summary)
    write_toml_flat(result_dir / "config_used.toml", config.to_dict())
    write_centerlines(result_dir, lines)
    save_cavity_plots(result_dir, rho, u, state.solid, lines)
    return result_dir, summary

