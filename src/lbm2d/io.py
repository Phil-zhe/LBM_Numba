from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np


def unique_result_dir(root: str | Path, label: str):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{label}_{stamp}"
    path.mkdir()
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_toml_flat(path, data):
    lines = []
    for key, value in data.items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        else:
            lines.append(f"{key} = {value}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_centerlines(result_dir, lines):
    x, v, y, u, *_ = lines
    for name, coord_name, value_name, coords, values in [
        ("centerline_u.csv", "y", "u_over_U_lid", y, u),
        ("centerline_v.csv", "x", "v_over_U_lid", x, v),
    ]:
        with (Path(result_dir) / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([coord_name, value_name])
            writer.writerows(zip(coords, values))


def save_cavity_plots(result_dir, rho, u, solid, lines):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(result_dir) / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = Path(result_dir)
    speed = np.sqrt(u[0] ** 2 + u[1] ** 2)
    masked_speed = np.ma.array(speed, mask=solid)
    for filename, field, title, cmap in [
        ("velocity_magnitude.png", masked_speed, "Velocity magnitude", "viridis"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(field, origin="lower", cmap=cmap)
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(result_dir / filename, dpi=160)
        plt.close(fig)

    duydx = np.gradient(u[1], axis=1)
    duxdy = np.gradient(u[0], axis=0)
    vort = np.ma.array(duydx - duxdy, mask=solid)
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = max(float(np.max(np.abs(vort))), 1e-12)
    image = ax.imshow(vort, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim)
    fig.colorbar(image, ax=ax)
    ax.set_title("Vorticity")
    fig.tight_layout()
    fig.savefig(result_dir / "vorticity.png", dpi=160)
    plt.close(fig)

    ygrid, xgrid = np.mgrid[0:u.shape[1], 0:u.shape[2]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.streamplot(xgrid, ygrid, u[0], u[1], density=1.3, color=speed, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("Streamlines")
    fig.tight_layout()
    fig.savefig(result_dir / "streamlines.png", dpi=160)
    plt.close(fig)

    x, v, y, uu, gx, gv, gy, gu = lines
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(uu, y, label="LBM")
    axes[0].scatter(gu, gy, label="Ghia", s=14)
    axes[0].set(xlabel="u/U_lid", ylabel="y", title="Vertical centerline")
    axes[1].plot(x, v, label="LBM")
    axes[1].scatter(gx, gv, label="Ghia", s=14)
    axes[1].set(xlabel="x", ylabel="v/U_lid", title="Horizontal centerline")
    for ax in axes:
        ax.grid(True, alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "cavity_ghia_comparison.png", dpi=160)
    plt.close(fig)
