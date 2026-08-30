# LBM2D: NumPy CPU and Numba-CUDA-MLIR GPU solver

This project implements a two-dimensional, isothermal, weakly compressible
D2Q9 lattice-Boltzmann solver. The production CPU backend is vectorized NumPy;
the production GPU backend uses explicit kernels imported only as
`from numba_cuda_mlir import cuda`. It does not use Numba CPU JIT, traditional
Numba-CUDA, PyTorch, CuPy, or implicit host-array kernel transfers.

## Numerical contract

Arrays use `[q,y,x]` for distributions and `[component,y,x]` for velocity.
Directions are `(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)`
with opposite indices `[0,3,4,1,2,7,8,5,6]`. The canonical equilibrium and BGK
collision are

$$f_i^{eq}=w_i\rho[1+3e_i\cdot u+\tfrac92(e_i\cdot u)^2-\tfrac32u\cdot u],$$

$$f_i^*=f_i-\omega(f_i-f_i^{eq}),\qquad \tau=\tfrac12+3\nu,\quad\omega=1/\tau.$$

Each step computes macroscopic fields and collision, performs pull streaming
with halfway bounce-back, applies any open-boundary reconstruction, swaps
`f/f_next`, and invalidates the macroscopic cache. `get_fields()` refreshes from
the current post-swap `f`. Solid placeholders are deterministically
`f=f_post=f_next=0`, `rho=1`, `u=0`.

The cavity uses stationary outer-wall nodes, a moving top wall excluding the
stationary corners, and physical length `L=ny-2`. The cylinder representation
uses a periodic-y minimum-image circular mask, stationary halfway bounce-back,
left Zou–He velocity inlet, and right fixed-density/zero-normal-gradient
tangential-velocity Zou–He outlet. That outlet can reflect unsteady wakes.

GPU field arrays, masks, wall velocity, lattice constants, and signed source
index tables are allocated once and remain on the device during stepping. The
source tables avoid unsigned `cuda.grid` subtraction for negative lattice
directions in Numba-CUDA-MLIR 0.5.0. Compatibility memory APIs are isolated in
`device_adapter.py`; migration to `cuda.core` is a later controlled option.

## Installation and commands

```powershell
F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m pip install -e ".[dev,benchmark]"
F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m lbm2d.cli inspect
F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m pytest -q

F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m lbm2d.cli run `
  --config configs/cavity_re100.toml --backend numpy-cpu --mode full
F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m lbm2d.cli run `
  --config configs/cavity_re100.toml --backend numba-cuda-mlir --mode full
F:\miniconda3\envs\lbm_cuda_mlir311\python.exe -m lbm2d.cli benchmark `
  --sizes 129 257 513 --timed-steps 2000 --repeats 5
```

`inspect` records Python/package versions, CUDA availability, driver/GPU/CC,
thread-pool information and a real vector-add kernel. Full/validation runs save
fields, diagnostics, normalized centerlines, Ghia comparison and flow images
under unique `results/` directories. Benchmark timing excludes compilation,
initial allocation/copy, diagnostics, plotting and I/O; every repeat restores
the same canonical state, and GPU timing is explicitly synchronized.

## Validation and limitations

Ghia centerlines use halfway-wall coordinates `(index-0.5)/(n-2)`, interpolate
between numerical centerlines, and add exact physical wall endpoints before
interpolation to reference locations. See `RESULTS.md` for actual values and
the explicitly recorded long-time density/mass acceptance failures. Results are
never clipped, renormalized, reset, or silently switched to another backend.

Limitations: two-dimensional, single GPU, BGK/SRT, weakly compressible and
isothermal; no turbulence, multiphase, thermal, MPI, or multi-GPU model. The
cylinder is a staircase geometry. The NumPy baseline is not an optimized
compiled CPU kernel. Long unsteady CPU/GPU solutions can phase-drift. Cylinder
execution and Strouhal analysis were intentionally not run in this stage.

