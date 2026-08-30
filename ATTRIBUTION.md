# Attribution

The project organization and separation of lattice, backend, case and
validation concerns were informed by the public `Malyadeep/pylabolt` project,
as named in the supplied specification. No PyLaBolt source file or code fragment
was copied; this implementation was written independently against the numerical
contract in the supplied prompt. The reference repository license could not be
reliably retrieved during this run, so no license assertion is made here and it
must be checked at the exact commit before any future code reuse.

The cavity tables are transcribed from U. Ghia, K. N. Ghia and C. T. Shin,
“High-Re solutions for incompressible flow using the Navier-Stokes equations and
a multigrid method,” *Journal of Computational Physics* 48 (1982), 387–411.
Variables and transcription are documented in `reference/README.md`.

Numba-CUDA-MLIR is an NVIDIA dependency distributed under Apache-2.0 according
to the installed 0.5.0 package metadata. Mentioning it does not imply NVIDIA
endorsement of this solver.

