from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib

from .lattice import CS, viscosity_parameters


@dataclass(frozen=True)
class CavityConfig:
    case: str = "cavity"
    nx: int = 257
    ny: int = 257
    re: float = 100.0
    lid_velocity: float = 0.05
    precision: str = "float32"
    max_steps: int = 150000
    minimum_steps: int = 10000
    check_interval: int = 500
    diagnostic_interval: int = 500
    convergence_tolerance: float = 1.0e-6
    consecutive_converged_checks: int = 5
    checkpoint_interval: int = 10000

    def __post_init__(self):
        if self.case != "cavity":
            raise ValueError("CavityConfig requires case='cavity'")
        if self.nx < 5 or self.ny < 5:
            raise ValueError("cavity dimensions must be at least 5")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        if min(self.max_steps, self.check_interval, self.diagnostic_interval) <= 0:
            raise ValueError("step counts and intervals must be positive")
        if self.minimum_steps < 0 or self.minimum_steps > self.max_steps:
            raise ValueError("minimum_steps must be in [0, max_steps]")
        if self.consecutive_converged_checks < 1:
            raise ValueError("consecutive_converged_checks must be positive")
        _, tau, omega = viscosity_parameters(self.re, self.lid_velocity, self.length)
        if self.ma_ref > 0.12:
            raise ValueError(f"Ma_ref={self.ma_ref:.6g} exceeds hard limit 0.12")
        if not (tau > 0.5 and 0 < omega < 2):
            raise ValueError("invalid relaxation parameters")

    @property
    def length(self) -> int:
        return self.ny - 2

    @property
    def nu_tau_omega(self):
        return viscosity_parameters(self.re, self.lid_velocity, self.length)

    @property
    def ma_ref(self) -> float:
        return self.lid_velocity / CS

    def to_dict(self):
        d = asdict(self)
        d.update(zip(("nu", "tau", "omega"), self.nu_tau_omega))
        d.update(length=self.length, ma_ref=self.ma_ref)
        return d


def load_cavity_config(path: str | Path) -> CavityConfig:
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    return CavityConfig(**data)


@dataclass(frozen=True)
class CylinderConfig:
    case: str = "cylinder"
    nx: int = 801
    ny: int = 320
    re: float = 100.0
    inlet_velocity: float = 0.05
    outlet_density: float = 1.0
    cylinder_diameter: float = 40.0
    cylinder_center_x: float = 200.0
    cylinder_center_y: float = 160.0
    precision: str = "float32"
    initial_steps: int = 80000
    maximum_steps: int = 120000
    extension_chunk_steps: int = 20000
    transient_steps: int = 20000
    analysis_window_steps: int = 60000
    sample_interval: int = 5
    flux_sample_interval: int = 50
    diagnostic_interval: int = 500
    probe_x_offset_d: float = 4.0
    probe_y_offset_d: float = 0.5
    initial_perturbation_amplitude: float = 1.0e-3
    initial_perturbation_phase: float = 0.7853981633974483
    periodic_y: bool = True
    checkpoint_interval: int = 10000

    def __post_init__(self):
        if self.case != "cylinder" or not self.periodic_y:
            raise ValueError("cylinder requires case='cylinder' and periodic_y=true")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        _, tau, omega = self.nu_tau_omega
        if not (tau > 0.5 and 0 < omega < 2) or self.ma_ref > 0.12:
            raise ValueError("invalid cylinder relaxation or Mach parameters")
        if not (0 <= self.probe_x < self.nx and 0 <= self.probe_y < self.ny):
            raise ValueError("derived probe lies outside the domain")

    @property
    def nu_tau_omega(self):
        return viscosity_parameters(self.re, self.inlet_velocity, self.cylinder_diameter)

    @property
    def ma_ref(self):
        return self.inlet_velocity / CS

    @property
    def probe_x(self):
        return self.cylinder_center_x + self.probe_x_offset_d * self.cylinder_diameter

    @property
    def probe_y(self):
        return self.cylinder_center_y + self.probe_y_offset_d * self.cylinder_diameter

    def to_dict(self):
        d = asdict(self)
        d.update(zip(("nu", "tau", "omega"), self.nu_tau_omega))
        d.update(ma_ref=self.ma_ref, probe_x=self.probe_x, probe_y=self.probe_y)
        return d


def load_cylinder_config(path: str | Path) -> CylinderConfig:
    with Path(path).open("rb") as handle:
        return CylinderConfig(**tomllib.load(handle))
