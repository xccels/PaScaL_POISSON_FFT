#!/usr/bin/env python3
"""Validation and benchmark of the wrapper library against the analytic solution.

Solves the 3D Poisson equation on the GPU through pascal_poisson, with the same
test problem the Fortran example uses, and reports the RMS error against the
exact solution.  All arrays are device-resident (CuPy); nothing but device
addresses and MPI handles crosses into Fortran.

Test problem, on [0, L]^3 with periodic x and y and Neumann z:

    p_exact = cos(PI x) cos(PI y) cos(PI z)
    rhs     = lambda_discrete * p_exact,
    lambda_discrete = -sum 4 sin(PI h_i / 2)^2 / h_i^2

The coefficient is the eigenvalue of the *discrete* second-order Laplacian, not
the continuous -3 PI^2.  p_exact is then an exact eigenvector of the discretised
problem, so a correct solve must reproduce it to round-off.  With the continuous
coefficient the best possible answer would still sit ~1.8e-5 away at 256^3, and
a real failure would be hard to tell from discretisation error.  The test stays
valid on any grid because cos(PI x) has period 2 = L and zero slope at z = 0, L.

Reference values from the Fortran example (01_Fortran, 256^3, A100):
    np=1 : RMS = 6.4000356853520139E-014
    np=2 : RMS = 6.3592055150029353E-014

Usage:
    mpirun -n 1 python poisson3d_fft.py
    POISSON_NX=256 POISSON_NY=256 POISSON_NZ=512 POISSON_NP=2,2,4 \
        mpirun -n 16 python poisson3d_fft.py

Environment:
    POISSON_N            cells per direction, default for all three  (default 256)
    POISSON_NX/NY/NZ     per-direction cell counts                   (default POISSON_N)
    POISSON_NP           process grid "np1,np2,np3"                  (default: split z)
    POISSON_ITERS        timed solves after warm-up                  (default 20)
    POISSON_WARMUP       untimed solves before timing                (default 5)

Rank 0 prints one machine-readable line for the benchmark driver:

    BENCH,<np>,<np1>x<np2>x<np3>,<nx>,<ny>,<nz>,<iters>,<solve_s>,
          <comp_s>,<comm_s>,<split_total_s>,<closure>,<rms>,<OK|BAD>

<comp_s>/<comm_s> are measured inside the solver and require POISSON_TIMER=1;
they are zero otherwise.  <closure> is (comp+comm)/total and should be 1.

Run on A100.  The solver silently returns a zero field on V100.
"""
import math
import os
import sys

# pascal_poisson must come before mpi4py: it loads the solver's shared library,
# and the OpenACC runtime inside it finds no GPU if libmpi is loaded first.
# The module raises ImportError if this order is broken.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pascal_poisson as pp                                     # noqa: E402

import cupy as cp                                               # noqa: E402
import numpy as np                                              # noqa: E402
from mpi4py import MPI                                          # noqa: E402

REFERENCE = {1: 6.4000356853520139e-14, 2: 6.3592055150029353e-14}
# Generous next to the 6e-14 reference, five orders tighter than the 1.8e-5 that
# a continuous-coefficient right-hand side would leave behind.
TOL = 1.0e-10

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size

n_default = int(os.environ.get("POISSON_N", 256))
nx = int(os.environ.get("POISSON_NX", n_default))
ny = int(os.environ.get("POISSON_NY", n_default))
nz = int(os.environ.get("POISSON_NZ", n_default))
iters = int(os.environ.get("POISSON_ITERS", 20))
warmup = int(os.environ.get("POISSON_WARMUP", 5))
if "POISSON_NP" in os.environ:
    nproc = tuple(int(v) for v in os.environ["POISSON_NP"].split(","))
else:
    nproc = (1, 1, size)                    # z-split, as PARA_INPUT_<size>.dat does
L = (2.0, 2.0, 2.0)
pbc = (True, True, False)                   # periodic x and y, Neumann z

dev = pp.bind_gpu()                         # before any CuPy allocation
dec = pp.Decomposition(n=(nx, ny, nz), L=L, pbc=pbc, nproc=nproc)

if rank == 0:
    print(f"grid {nx}x{ny}x{nz}   L {L}   pbc {pbc}   nproc {nproc}")
    print(f"subdomain (n1msub,n2msub,n3msub) = {dec.rhs_shape}   p shape = {dec.p_shape}")
print(f"  rank {rank}: GPU {dev}  i {dec.ista}..{dec.iend}  "
      f"j {dec.jsta}..{dec.jend}  k {dec.ksta}..{dec.kend}", flush=True)

# --- test problem -----------------------------------------------------------
hx, hy, hz = L[0] / nx, L[1] / ny, L[2] / nz         # uniform grid
lam = -(4.0 * math.sin(0.5 * math.pi * hx) ** 2 / hx ** 2 +
        4.0 * math.sin(0.5 * math.pi * hy) ** 2 / hy ** 2 +
        4.0 * math.sin(0.5 * math.pi * hz) ** 2 / hz ** 2)
if rank == 0:
    print(f"discrete eigenvalue {lam:.10f}   continuous -3*pi^2 {-3 * math.pi ** 2:.10f}")

xc, yc, zc = (cp.asarray(a) for a in dec.cell_centres())
p_exact = cp.cos(cp.pi * xc) * cp.cos(cp.pi * yc) * cp.cos(cp.pi * zc)

rhs = dec.empty_rhs()
p = dec.zeros_p()
i1, i2, i3 = dec.n1msub, dec.n2msub, dec.n3msub


def fill_rhs():
    """The solver consumes rhs as scratch, so rebuild it before every solve."""
    rhs[...] = lam * p_exact


rms = float("nan")
t_max = float("nan")
t_comp = t_comm = t_tot = closure = 0.0
plan = pp.PoissonPlan(dec)
try:
    # --- correctness ---------------------------------------------------------
    fill_rhs()
    plan.solve(rhs, p)
    plan.sync()

    diff = p[1:i1 + 1, 1:i2 + 1, 1:i3 + 1] - p_exact
    err_global = comm.allreduce(float(cp.sum(diff * diff)), op=MPI.SUM)
    rms = math.sqrt(err_global / (nx * ny * nz))

    # --- timing --------------------------------------------------------------
    for _ in range(warmup):
        fill_rhs()
        plan.solve(rhs, p)
    plan.sync()
    # Warm-up carries cuFFT plan creation and cuDecomp autotuning; drop it so the
    # computation/communication split reflects a steady-state solve.
    plan.timer_clear()
    comm.Barrier()

    times = np.empty(iters)
    for it in range(iters):
        # Rebuilding the right-hand side is outside the timer, as in main.f90.
        fill_rhs()
        plan.sync()
        t0 = MPI.Wtime()
        plan.solve(rhs, p)
        plan.sync()
        times[it] = MPI.Wtime() - t0
    t_max = comm.allreduce(float(times.mean()), op=MPI.MAX)

    # Computation/communication split, measured inside the solver (POISSON_TIMER=1).
    # Rank-mean per solve, matching the convention of the wall-clock number above.
    r_comp, r_comm, r_tot, nsolve = plan.timer_read()
    if nsolve > 0:
        t_comp = comm.allreduce(r_comp, op=MPI.SUM) / size / nsolve
        t_comm = comm.allreduce(r_comm, op=MPI.SUM) / size / nsolve
        t_tot = comm.allreduce(r_tot, op=MPI.SUM) / size / nsolve
        closure = (t_comp + t_comm) / t_tot if t_tot > 0 else 0.0
finally:
    plan.destroy()

ok = rms < TOL
if rank == 0:
    ref = REFERENCE.get(size) if nx == ny == nz == 256 else None
    print()
    print(f"RMS error against analytic solution: {rms:.16e}")
    if ref is not None:
        print(f"Fortran reference (np={size})      : {ref:.16e}")
        print(f"ratio python/fortran               : {rms / ref:.4f}")
    print(f"mean solve time (rank-max)         : {t_max * 1e3:.3f} ms")
    if t_tot > 0:
        print(f"  (a) computation                  : {t_comp * 1e3:.3f} ms")
        print(f"  (b) communication                : {t_comm * 1e3:.3f} ms  "
              f"({100 * t_comm / t_tot:.1f} %)")
        print(f"  (c) total (instrumented)         : {t_tot * 1e3:.3f} ms  "
              f"closure {closure:.5f}")
    else:
        print("  computation/communication split  : off (set POISSON_TIMER=1)")
    print()
    print(f"BENCH,{size},{nproc[0]}x{nproc[1]}x{nproc[2]},{nx},{ny},{nz},"
          f"{iters},{t_max:.6e},{t_comp:.6e},{t_comm:.6e},{t_tot:.6e},"
          f"{closure:.5f},{rms:.6e},{'OK' if ok else 'BAD'}")
    print("PASS" if ok else "FAIL")

sys.exit(0 if ok else 1)
