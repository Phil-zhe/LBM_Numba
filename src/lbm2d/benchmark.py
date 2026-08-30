from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
import numpy as np

from .cases.cavity import make_solver
from .config import CavityConfig
from .diagnostics import field_diagnostics
from .io import unique_result_dir, write_json


def _warm(solver, minimum_steps, minimum_seconds):
    count = 0
    start = time.perf_counter()
    while count < minimum_steps or time.perf_counter() - start < minimum_seconds:
        solver.step()
        count += 1
    if hasattr(solver, "synchronize"):
        solver.synchronize()
    return count, time.perf_counter() - start


def benchmark_cavity(sizes=(129, 257, 513), timed_steps=2000, repeats=5,
                     warmup_steps=100, warmup_min_seconds=1.0,
                     results_root="results", seed=20260828):
    rows = []
    detail = {"seed": seed, "sizes": list(sizes), "timed_steps": timed_steps,
              "repeats": repeats, "warmup_steps": warmup_steps,
              "warmup_min_seconds": warmup_min_seconds, "runs": []}
    rng = random.Random(seed)
    for n in sizes:
        cfg = CavityConfig(nx=n, ny=n, max_steps=timed_steps, minimum_steps=0,
                           check_interval=timed_steps, diagnostic_interval=timed_steps)
        cpu, cpu_state = make_solver(cfg, "numpy-cpu")
        gpu, gpu_state = make_solver(cfg, "numba-cuda-mlir")
        canonical = cpu.copy_current_f()

        # Trigger all GPU specializations and record combined first-call latency.
        cold_start = time.perf_counter()
        gpu.step()
        gpu.synchronize()
        cold_start = time.perf_counter() - cold_start
        cpu_first = time.perf_counter()
        cpu.step()
        cpu_first = time.perf_counter() - cpu_first
        cpu.restore_f(canonical)
        gpu.restore_f(canonical)
        cpu_warm_count, cpu_warm_elapsed = _warm(cpu, warmup_steps, warmup_min_seconds)
        cpu.restore_f(canonical)
        gpu_warm_count, gpu_warm_elapsed = _warm(gpu, warmup_steps, warmup_min_seconds)
        gpu.restore_f(canonical)
        gpu.synchronize()

        order = [backend for _ in range(repeats) for backend in ("numpy-cpu", "numba-cuda-mlir")]
        rng.shuffle(order)
        times = {"numpy-cpu": [], "numba-cuda-mlir": []}
        for backend in order:
            solver = cpu if backend == "numpy-cpu" else gpu
            solver.restore_f(canonical)
            if hasattr(solver, "synchronize"):
                solver.synchronize()
            start = time.perf_counter()
            for _ in range(timed_steps):
                solver.step()
            if hasattr(solver, "synchronize"):
                solver.synchronize()
            elapsed = time.perf_counter() - start
            times[backend].append(elapsed)
            rho, u = solver.get_fields()
            d = field_diagnostics(rho, u, cpu_state.solid, float((~cpu_state.solid).sum()))
            if d["nonfinite_count"] or d["min_rho"] <= 0:
                raise RuntimeError(f"invalid benchmark state: {backend} {n}: {d}")

        med_cpu = float(np.median(times["numpy-cpu"]))
        med_gpu = float(np.median(times["numba-cuda-mlir"]))
        nfluid = int((~cpu_state.solid).sum())
        for backend, solver_times in times.items():
            median = float(np.median(solver_times))
            row = {
                "grid": f"{n}x{n}", "nx": n, "ny": n, "precision": cfg.precision,
                "backend": backend, "timed_steps": timed_steps,
                "repeat_times_seconds": solver_times, "median_seconds": median,
                "min_seconds": min(solver_times), "max_seconds": max(solver_times),
                "mlups_all": n * n * timed_steps / (1e6 * median),
                "mlups_fluid": nfluid * timed_steps / (1e6 * median),
                "speedup_mlir_over_numpy": med_cpu / med_gpu,
            }
            rows.append(row)
        detail["runs"].append({
            "grid": f"{n}x{n}", "repeat_order": order, "times": times,
            "gpu_cold_start_first_call_time": cold_start,
            "cpu_first_run_time": cpu_first,
            "cpu_warmup": {"steps": cpu_warm_count, "seconds": cpu_warm_elapsed},
            "gpu_warmup": {"steps": gpu_warm_count, "seconds": gpu_warm_elapsed},
        })

    result_dir = unique_result_dir(results_root, "benchmark_cavity")
    write_json(result_dir / "benchmark_results.json", {"summary": rows, **detail})
    with (result_dir / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [k for k in rows[0] if k != "repeat_times_seconds"] + ["repeat_times_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["repeat_times_seconds"] = json.dumps(out["repeat_times_seconds"])
            writer.writerow(out)
    _plot_benchmark(result_dir, rows)
    return result_dir, rows


def _plot_benchmark(result_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grids = sorted({r["nx"] for r in rows})
    for filename, key, ylabel in [
        ("benchmark_mlups.png", "mlups_all", "MLUPS (all nodes)"),
        ("benchmark_speedup.png", "speedup_mlir_over_numpy", "GPU / NumPy CPU speedup"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        if key == "mlups_all":
            for backend in ("numpy-cpu", "numba-cuda-mlir"):
                values = [next(r[key] for r in rows if r["nx"] == n and r["backend"] == backend) for n in grids]
                ax.plot(grids, values, marker="o", label=backend)
            ax.legend()
        else:
            values = [next(r[key] for r in rows if r["nx"] == n) for n in grids]
            ax.bar([str(n) for n in grids], values)
        ax.set(xlabel="Grid size", ylabel=ylabel)
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        fig.savefig(Path(result_dir) / filename, dpi=160)
        plt.close(fig)

