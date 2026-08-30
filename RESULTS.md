# Actual results (2026-08-28)

Only measured values are listed. Cylinder flow and cylinder performance were
not executed at the user's request.

## Environment

- Windows 10 x64; Intel Core i7-10875H; 32 GB RAM
- Python 3.11.16; NumPy 2.4.6; OpenBLAS 0.3.31, reported pool size 16
- Numba-CUDA-MLIR 0.5.0; cuda-core 1.1.1; cuda-bindings 12.9.7;
  CUDA toolkit 12.9.2.0
- NVIDIA GeForce RTX 2060 6 GB, compute capability 7.5, driver 591.74
- CUDA available: yes; minimal kernel: pass; 500-launch LBM regression: pass
- Precision: float32

## Tests

Installed-environment run: **15 passed**. Covered dependency/AST contracts,
D2Q9 isotropy and equilibrium moments, NumPy/reference stepping, moving-wall
sign, closed mass smoke, left/right Zou–He moments, uniform open-boundary CPU
and CPU/GPU behavior, checkpoint/resume equivalence, 1/5/20-step full-field
CPU/GPU consistency, 500 repeated GPU launches, and cylinder geometry/initial
perturbation construction. Small-grid under-occupancy warnings are expected.

## Cavity validation/full runs

| Metric | NumPy CPU, 129² | MLIR GPU, 129² | MLIR GPU, 257² |
|---|---:|---:|---:|
| steps | 150000 | 150000 | 150000 |
| full-loop wall time (s) | 213.845 | 29.702 | 112.943 |
| converged | false | false | false |
| final residual | 5.494e-6 | 5.629e-6 | 1.160e-6 |
| rho deviation peak | 0.047512 | 0.047426 | 0.062859 |
| Ma peak | 0.084529 | 0.084529 | 0.085570 |
| mass drift | 0.005625 | 0.005542 | 0.001958 |
| density range | 0.973664–1.047512 | 0.973584–1.047426 | 0.953529–1.062859 |
| Ghia u RMSE | 0.002581 | 0.002588 | 0.002469 |
| Ghia v RMSE | 0.004588 | 0.004602 | 0.004687 |
| Ghia u max error | 0.005400 | 0.005427 | 0.005191 |
| Ghia v max error | 0.008844 | 0.008871 | 0.009065 |

The same-grid 129² full-loop wall-time ratio is **7.20×** GPU/NumPy CPU. CPU
and GPU final-centerline RMSE differences were `1.731e-5` (u) and `1.858e-5`
(v); maximum centerline differences were `3.056e-5` and `3.146e-5`. Full-field
maximum velocity difference was `1.940e-6`.

Ghia and Mach thresholds pass. Strict convergence and mass-drift thresholds do
not pass. At 257², density deviation also exceeds 0.05; its extrema occur at
the two fluid nodes immediately below the stationary top corners. No clipping,
renormalization, distribution reset, viscosity change or node exclusion was
used to hide these failures. A 257² NumPy full run was not executed because the
measured 2000-step rate predicts about 18 minutes; the required same-grid fixed
step comparison was executed.

## Fixed-step benchmark

Each entry used 2000 steps, five repeats, float32, a shared canonical initial
state, fixed-seed interleaved order, at least 100 warmup steps and at least one
second accumulated warmup. Values are all-grid MLUPS.

| Grid | Backend | repeat seconds | median s | MLUPS | speedup |
|---|---|---|---:|---:|---:|
|129²|NumPy CPU|3.909, 3.847, 3.704, 3.779, 3.757|3.779|8.807|5.52×|
|129²|MLIR GPU|0.667, 0.684, 0.684, 0.705, 0.703|0.684|48.632|5.52×|
|257²|NumPy CPU|15.436, 14.986, 14.751, 14.413, 14.508|14.751|8.955|7.42×|
|257²|MLIR GPU|2.004, 1.961, 2.024, 1.972, 1.987|1.987|66.484|7.42×|
|513²|NumPy CPU|83.504, 85.454, 70.707, 61.149, 61.431|70.707|7.444|44.12×|
|513²|MLIR GPU|7.202, 7.456, 1.603, 1.597, 1.243|1.603|328.445|44.12×|

The large 513² variation is retained in full; the median, not fastest/slowest
pairing, defines speedup. First GPU call at 129² was 0.948 s and includes
context/JIT/module work, not pure compile time. `end_to_end_no_compile_time` was
not separately executed and is reported as **not executed**.

## Result locations

- Same-grid CPU: `results/cavity_numpy-cpu_20260828_202807_426032`
- Same-grid GPU: `results/cavity_numba-cuda-mlir_20260828_203101_245510`
- 257² GPU: `results/cavity_numba-cuda-mlir_20260828_201236_032434`
- Benchmark: `results/benchmark_cavity_20260828_202404_504828`
- Cylinder: **not executed**

