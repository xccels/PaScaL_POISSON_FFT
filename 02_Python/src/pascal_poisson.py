"""Python interface to the device-resident CUDA FFT Poisson solver of PaScaL_POISSON_FFT.

Loads libpoisson_capi.so (ctypes) and exposes the solver through two classes:

    Decomposition   pure Python.  Builds the MPI cartesian topology, the block
                    decomposition, the grid metrics, and the MPI derived
                    datatypes for the C<->I and C<->J transposes.  This is what
                    examples/mpi_topology.f90 and examples/mpi_subdomain.f90 do
                    in the Fortran reference.  It lives here rather than in the
                    example because the solver, not the problem, is what fixes
                    these quantities: every caller needs exactly the same ones.
    PoissonPlan     thin ctypes layer over the bind(C) wrapper.  Owns everything
                    that touches the fft_poisson module state: plan creation,
                    cuDecomp and static-TDMA initialisation, the solve, teardown.

Right-hand side and solution are CuPy arrays; only their device addresses cross
the language boundary, so the data never leaves the GPU.  MPI communicators and
derived datatypes cross as Fortran integer handles, i.e. what mpi4py's py2f()
returns.  Python and the shared library must therefore be built against the same
MPI, or those handles refer to different tables on the two sides.

Array convention: dtype float64, Fortran order (order='F').
    rhs   (n1msub, n2msub, n3msub)      overwritten by solve(); the solver reuses
                                        it as working storage
    p     (n1sub+1, n2sub+1, n3sub+1)   solution, one ghost layer included

Typical use (one MPI rank per GPU):

    from mpi4py import MPI
    import pascal_poisson as pp

    pp.bind_gpu()                       # before any CuPy allocation
    dec  = pp.Decomposition(n=(256, 256, 256), L=(2.0, 2.0, 2.0),
                            pbc=(True, True, False), nproc=(1, 1, 2))
    plan = pp.PoissonPlan(dec)
    rhs, p = dec.empty_rhs(), dec.zeros_p()
    ...                                 # fill rhs
    plan.solve(rhs, p)
    plan.destroy()

Run on A100.  The same solver silently returns a zero field on V100; see the
comment in 01_Fortran/run_reference_rms.slurm.
"""
import ctypes
import os
import sys

# ---------------------------------------------------------------------------
# Load the shared library BEFORE mpi4py -- the order is load-bearing.
#
# Importing mpi4py first pulls in HPC-X libmpi and its CUDA components, and the
# NVHPC OpenACC runtime inside this library then enumerates zero devices:
# acc_get_num_devices() returns 0, and the first !$acc region in the solver dies
# with "No accelerator device found for cudafor_acc_malloc call" during plan
# creation.  Measured on a 2-GPU node: library-then-mpi4py gives 2 devices,
# mpi4py-then-library gives 0.  CuPy has no such effect; only mpi4py does.
#
# PaScaL_TDMA's wrapper is immune because that library is pure CUDA Fortran with
# no OpenACC directives, so it has no OpenACC runtime to lose.
# ---------------------------------------------------------------------------
if "mpi4py" in sys.modules:
    raise ImportError(
        "pascal_poisson must be imported before mpi4py: the solver's OpenACC "
        "runtime finds no GPU when libmpi is loaded first, and plan creation "
        "then aborts with 'No accelerator device found'. "
        "Move 'import pascal_poisson' above 'from mpi4py import MPI'.")

_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "lib", "libpoisson_capi.so")
_lib = ctypes.CDLL(_SO)

import numpy as np                                              # noqa: E402
import cupy as cp                                               # noqa: E402
from mpi4py import MPI                                          # noqa: E402

# argtypes are not decoration: without them ctypes passes Python ints as 64-bit
# values while the wrapper reads 32-bit integer(c_int), and arguments arrive
# corrupted.
_INT_P = ctypes.POINTER(ctypes.c_int)
_I32ARR = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

_lib.poisson_para_range.restype = None
_lib.poisson_para_range.argtypes = [ctypes.c_int] * 4 + [_INT_P] * 2
# 21 ints, 3 doubles (L1..L3), then 9 ints (periodic flags + two thread blocks)
_lib.poisson_plan_create.restype = None
_lib.poisson_plan_create.argtypes = ([ctypes.c_int] * 21 + [ctypes.c_double] * 3 +
                                     [ctypes.c_int] * 9)
_lib.poisson_cudecomp_init.restype = None
_lib.poisson_cudecomp_init.argtypes = [ctypes.c_int] * 2
_lib.poisson_tdma_static_init.restype = None
_lib.poisson_tdma_static_init.argtypes = ([ctypes.c_int] * 3 +
                                          [ctypes.c_void_p] * 2 + [ctypes.c_int])
# 6 device addresses, 4 ints, 12 handle arrays, 7 ints
_lib.poisson_solve.restype = None
_lib.poisson_solve.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 +
                               [_I32ARR] * 12 + [ctypes.c_int] * 7)
_lib.poisson_sync.restype = None
_lib.poisson_sync.argtypes = [_INT_P]
_lib.poisson_timer_clear.restype = None
_lib.poisson_timer_clear.argtypes = []
_lib.poisson_timer_read.restype = None
_lib.poisson_timer_read.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [_INT_P]
_lib.poisson_destroy.restype = None
_lib.poisson_destroy.argtypes = []


def device_count():
    """Number of visible CUDA devices."""
    return cp.cuda.runtime.getDeviceCount()


def bind_gpu(comm=MPI.COMM_WORLD, device=None):
    """Bind this rank to one GPU.  Call before allocating any CuPy array.

    CuPy's device selection is enough for the whole stack: the solver's CUDA
    Fortran kernels and its 51 OpenACC regions both follow it.  Verified at np=1
    and np=2, where a mismatch between the two runtimes would have put the
    OpenACC regions on device 0 while the data sat on another GPU and corrupted
    the result; the RMS matched the Fortran reference instead.  Setting the
    OpenACC device explicitly on the Fortran side proved unnecessary.

    device defaults to the node-local rank, as in the Fortran example.  World
    rank modulo device count would collide whenever ranks are permuted for
    topology tests.
    """
    if device is None:
        local = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
        device = local.rank % max(1, device_count())
        local.Free()
    cp.cuda.Device(device).use()
    return device


def para_range(nsta, nend, nprocs, myrank):
    """Block-distribute [nsta, nend] over nprocs; returns the inclusive (first, last).

    Delegated to the library instead of reimplemented.  A one-off difference here
    would raise nothing; it would quietly shift which wavenumbers the solver
    works on.
    """
    a, b = ctypes.c_int(0), ctypes.c_int(0)
    _lib.poisson_para_range(nsta, nend, nprocs, myrank, ctypes.byref(a), ctypes.byref(b))
    return a.value, b.value


def _extent(nsta, nend, nprocs, myrank):
    a, b = para_range(nsta, nend, nprocs, myrank)
    return b - a + 1


def _devptr(a, shape, name):
    """Validate a CuPy array against the plan and return its device address."""
    if a.dtype != cp.float64:
        raise TypeError(f"{name} must be float64, got {a.dtype}")
    if a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} != expected {shape}")
    if not a.flags.f_contiguous:
        raise ValueError(f"{name} must be Fortran-contiguous (order='F')")
    return ctypes.c_void_p(a.data.ptr)


class _Comm1D:
    """One direction of the cartesian topology, mirroring cart_comm_1d."""

    def __init__(self, cart, remain):
        self.comm = cart.Sub(remain)
        self.rank = self.comm.rank
        self.size = self.comm.size
        # MPI_Cart_shift returns (source, dest); the Fortran names them west, east.
        self.west, self.east = self.comm.Shift(0, 1)

    @property
    def handle(self):
        return self.comm.py2f()


class Decomposition:
    """Grid, MPI topology and transpose datatypes for one Poisson problem.

    n     (n1m, n2m, n3m) global cell counts; all three must be even
    L     (L1, L2, L3) domain lengths, origin at 0
    pbc   periodicity per direction; False means Neumann
    nproc (np1, np2, np3) process grid; the product must equal comm.size
    """

    def __init__(self, n, L, pbc, nproc, comm=MPI.COMM_WORLD,
                 threads_tdma=(32, 8, 1), threads_fft=(1, 1, 1)):
        self.n1m, self.n2m, self.n3m = (int(v) for v in n)
        self.L1, self.L2, self.L3 = (float(v) for v in L)
        self.pbc = tuple(bool(v) for v in pbc)
        self.np1, self.np2, self.np3 = (int(v) for v in nproc)
        self.threads_tdma = tuple(int(v) for v in threads_tdma)
        self.threads_fft = tuple(int(v) for v in threads_fft)

        if any(v % 2 for v in (self.n1m, self.n2m, self.n3m)):
            raise ValueError("n1m, n2m and n3m must all be even")
        if self.np1 * self.np2 * self.np3 != comm.size:
            raise ValueError(f"nproc product {self.np1 * self.np2 * self.np3} "
                             f"!= comm size {comm.size}")
        # The C2I all-to-all needs at least two x-planes per (x, z) rank pair.
        if self.n1m < 2 * self.np1 * self.np3 - 2:
            raise ValueError("n1m must be >= 2*np1*np3 - 2 for the C2I exchange")

        self.n1, self.n2, self.n3 = self.n1m + 1, self.n2m + 1, self.n3m + 1

        self._build_topology(comm)
        self._build_subdomain()
        self._build_grid()
        self._build_transpose_metadata()
        self._build_datatypes()

    # -- topology ---------------------------------------------------------
    def _build_topology(self, comm):
        self.comm = comm
        self.cart = comm.Create_cart([self.np1, self.np2, self.np3],
                                     periods=list(self.pbc), reorder=False)
        self.c1 = _Comm1D(self.cart, [True, False, False])
        self.c2 = _Comm1D(self.cart, [False, True, False])
        self.c3 = _Comm1D(self.cart, [False, False, True])

    # -- block decomposition ----------------------------------------------
    def _build_subdomain(self):
        self.ista, self.iend = para_range(1, self.n1m, self.np1, self.c1.rank)
        self.jsta, self.jend = para_range(1, self.n2m, self.np2, self.c2.rank)
        self.ksta, self.kend = para_range(1, self.n3m, self.np3, self.c3.rank)

        self.n1sub = self.iend - self.ista + 2
        self.n2sub = self.jend - self.jsta + 2
        self.n3sub = self.kend - self.ksta + 2
        self.n1msub, self.n2msub, self.n3msub = (self.n1sub - 1, self.n2sub - 1,
                                                 self.n3sub - 1)

    # -- grid metrics ------------------------------------------------------
    def _halo_1d(self, arr, c1d, nsub, nmsub):
        """Exchange one value with each neighbour, as mpi_subdomain_ghostcell_1d.

        A transfer with MPI_PROC_NULL (a non-periodic edge) completes without
        touching the receive buffer, so starting it at zero leaves the edge with
        the zero that the Fortran allocate_and_init put there.
        """
        for src_idx, dst_idx, dest, source in ((nmsub, 0, c1d.east, c1d.west),
                                               (1, nsub, c1d.west, c1d.east)):
            sbuf = np.array([arr[src_idx]], dtype=np.float64)
            rbuf = np.zeros(1, dtype=np.float64)
            c1d.comm.Sendrecv(sendbuf=sbuf, dest=dest, sendtag=111,
                              recvbuf=rbuf, source=source, recvtag=111)
            arr[dst_idx] = rbuf[0]

    def _direction_grid(self, sta, end, nsub, nmsub, L, nm, periodic, c1d):
        """Uniform grid: face coordinates x, cell sizes dx, centre distances dmx.

        x is indexed 0..nsub with one ghost point at each end, so x[1] is the
        first interior face and x[i], x[i+1] bracket cell i.
        """
        k = np.arange(nsub + 1, dtype=np.float64)
        x = (k + sta - 2) * (L / nm)                      # x_start = 0

        dx = np.zeros(nsub + 1, dtype=np.float64)
        dx[1:nmsub + 1] = x[2:nmsub + 2] - x[1:nmsub + 1]
        self._halo_1d(dx, c1d, nsub, nmsub)
        if not periodic:
            if c1d.rank == 0:
                dx[0] = 0.0
            if c1d.rank == c1d.size - 1:
                dx[nsub] = 0.0

        dmx = np.zeros(nsub + 1, dtype=np.float64)
        dmx[1:nmsub + 1] = 0.5 * (dx[0:nmsub] + dx[1:nmsub + 1])
        self._halo_1d(dmx, c1d, nsub, nmsub)
        if not periodic and c1d.rank == c1d.size - 1:
            dmx[nsub] = 0.5 * (dx[nmsub] + dx[nsub])

        return x, dx, dmx

    def _build_grid(self):
        self.x1, self.dx1, self.dmx1 = self._direction_grid(
            self.ista, self.iend, self.n1sub, self.n1msub,
            self.L1, self.n1m, self.pbc[0], self.c1)
        self.x2, self.dx2, self.dmx2 = self._direction_grid(
            self.jsta, self.jend, self.n2sub, self.n2msub,
            self.L2, self.n2m, self.pbc[1], self.c2)
        self.x3, self.dx3, self.dmx3 = self._direction_grid(
            self.ksta, self.kend, self.n3sub, self.n3msub,
            self.L3, self.n3m, self.pbc[2], self.c3)

        # Only these four reach the solver.
        self.dmx1_d = cp.asarray(self.dmx1)
        self.dmx2_d = cp.asarray(self.dmx2)
        self.dmx3_d = cp.asarray(self.dmx3)
        self.dx3_d = cp.asarray(self.dx3)

        # Interior cell centres, for building a right-hand side or an exact solution.
        self.xc = 0.5 * (self.x1[1:self.n1msub + 1] + self.x1[2:self.n1msub + 2])
        self.yc = 0.5 * (self.x2[1:self.n2msub + 1] + self.x2[2:self.n2msub + 2])
        self.zc = 0.5 * (self.x3[1:self.n3msub + 1] + self.x3[2:self.n3msub + 2])

    # -- transpose index metadata ------------------------------------------
    def _build_transpose_metadata(self):
        np1, np2 = self.np1, self.np2
        r1, r2 = self.c1.rank, self.c2.rank

        # y-extent of this rank's x-aligned slab
        self.n2msub_Isub_jsta, self.n2msub_Isub_jend = para_range(
            1, self.n2msub, np1, r1)
        self.n2msub_Isub = self.n2msub_Isub_jend - self.n2msub_Isub_jsta + 1

        # Complex wavenumber count of the real-to-complex x transform
        self.h1p = self.n1m // 2 + 1
        a, b = para_range(1, self.h1p, np1, r1)
        self.h1psub = b - a + 1
        # The second split runs over [a, b], not over [1, h1psub]: the y-aligned
        # slab is carved out of this rank's own global wavenumber range, so the
        # start index it reports is global.
        self.h1psub_Jsub_ista, self.h1psub_Jsub_iend = para_range(a, b, np2, r2)
        self.h1psub_Jsub = self.h1psub_Jsub_iend - self.h1psub_Jsub_ista + 1

        self.n1msub_Jsub_ista, self.n1msub_Jsub_iend = para_range(
            1, self.n1msub, np2, r2)
        self.n1msub_Jsub = self.n1msub_Jsub_iend - self.n1msub_Jsub_ista + 1

        # Per-rank extents needed to place every subarray.  The *_all lists over
        # np2 are built from this rank's local sizes, matching the Fortran.
        self.n2msub_Isub_all = [_extent(1, self.n2msub, np1, i) for i in range(np1)]
        self.n1msub_all = [_extent(1, self.n1m, np1, i) for i in range(np1)]
        self.h1psub_all = [_extent(1, self.h1p, np1, i) for i in range(np1)]
        self.h1psub_Jsub_all = [_extent(1, self.h1psub, np2, i) for i in range(np2)]
        self.n1msub_Jsub_all = [_extent(1, self.n1msub, np2, i) for i in range(np2)]
        self.n2msub_all = [_extent(1, self.n2m, np2, i) for i in range(np2)]

    # -- derived datatypes --------------------------------------------------
    def _build_datatypes(self):
        # Keep the Datatype objects alive: freeing one invalidates its handle,
        # and the solver holds nothing but the integers.
        self._datatypes = []

        def sub(base, big, subsz, start):
            dt = base.Create_subarray(big, subsz, start, order=MPI.ORDER_FORTRAN)
            dt.Commit()
            self._datatypes.append(dt)
            return dt.py2f()

        def offset(lst, i):
            """Where rank i's slab starts: the extents of every earlier rank."""
            return int(sum(lst[:i]))

        R, C = MPI.DOUBLE_PRECISION, MPI.DOUBLE_COMPLEX
        n1msub, n2msub, n3msub = self.n1msub, self.n2msub, self.n3msub

        # C <-> I: cube split along y going out, x-aligned slab coming back.
        dble_C_in_C2I, dble_I_in_C2I, cplx_C_in_C2I, cplx_I_in_C2I = [], [], [], []
        for i in range(self.np1):
            dble_C_in_C2I.append(sub(
                R, [n1msub, n2msub, n3msub],
                [n1msub, self.n2msub_Isub_all[i], n3msub],
                [0, offset(self.n2msub_Isub_all, i), 0]))
            dble_I_in_C2I.append(sub(
                R, [self.n1m, self.n2msub_Isub, n3msub],
                [self.n1msub_all[i], self.n2msub_Isub, n3msub],
                [offset(self.n1msub_all, i), 0, 0]))
            cplx_C_in_C2I.append(sub(
                C, [self.h1psub, n2msub, n3msub],
                [self.h1psub, self.n2msub_Isub_all[i], n3msub],
                [0, offset(self.n2msub_Isub_all, i), 0]))
            cplx_I_in_C2I.append(sub(
                C, [self.h1p, self.n2msub_Isub, n3msub],
                [self.h1psub_all[i], self.n2msub_Isub, n3msub],
                [offset(self.h1psub_all, i), 0, 0]))

        # C <-> J: cube split along x going out, y-aligned slab coming back.
        cplx_C_in_C2J, cplx_J_in_C2J, dble_C_in_C2J, dble_J_in_C2J = [], [], [], []
        for i in range(self.np2):
            cplx_C_in_C2J.append(sub(
                C, [self.h1psub, n2msub, n3msub],
                [self.h1psub_Jsub_all[i], n2msub, n3msub],
                [offset(self.h1psub_Jsub_all, i), 0, 0]))
            cplx_J_in_C2J.append(sub(
                C, [self.h1psub_Jsub, self.n2m, n3msub],
                [self.h1psub_Jsub, self.n2msub_all[i], n3msub],
                [0, offset(self.n2msub_all, i), 0]))
            dble_C_in_C2J.append(sub(
                R, [n1msub, n2msub, n3msub],
                [self.n1msub_Jsub_all[i], n2msub, n3msub],
                [offset(self.n1msub_Jsub_all, i), 0, 0]))
            dble_J_in_C2J.append(sub(
                R, [self.n1msub_Jsub, self.n2m, n3msub],
                [self.n1msub_Jsub, self.n2msub_all[i], n3msub],
                [0, offset(self.n2msub_all, i), 0]))

        def i32(v):
            return np.asarray(v, dtype=np.int32)

        self.ddt_dble_C_in_C2I = i32(dble_C_in_C2I)
        self.ddt_dble_I_in_C2I = i32(dble_I_in_C2I)
        self.ddt_dble_J_in_C2J = i32(dble_J_in_C2J)
        self.ddt_dble_C_in_C2J = i32(dble_C_in_C2J)
        self.ddt_cplx_I_in_C2I = i32(cplx_I_in_C2I)
        self.ddt_cplx_C_in_C2I = i32(cplx_C_in_C2I)
        self.ddt_cplx_J_in_C2J = i32(cplx_J_in_C2J)
        self.ddt_cplx_C_in_C2J = i32(cplx_C_in_C2J)

        # MPI_Alltoallw counts: one datatype per partner, zero byte displacement.
        self.countsendI = np.ones(self.np1, dtype=np.int32)
        self.countdistI = np.zeros(self.np1, dtype=np.int32)
        self.countsendJ = np.ones(self.np2, dtype=np.int32)
        self.countdistJ = np.zeros(self.np2, dtype=np.int32)

    # -- array helpers ------------------------------------------------------
    @property
    def rhs_shape(self):
        return (self.n1msub, self.n2msub, self.n3msub)

    @property
    def p_shape(self):
        return (self.n1sub + 1, self.n2sub + 1, self.n3sub + 1)

    def empty_rhs(self):
        return cp.empty(self.rhs_shape, dtype=cp.float64, order="F")

    def zeros_p(self):
        return cp.zeros(self.p_shape, dtype=cp.float64, order="F")

    def cell_centres(self):
        """(xc, yc, zc) shaped to broadcast into an (n1msub, n2msub, n3msub) array."""
        return (self.xc[:, None, None], self.yc[None, :, None], self.zc[None, None, :])


class PoissonPlan:
    """The solver: plan creation, initialisation, solve, teardown.

    fft_poisson keeps its plan in a single module variable, so there is exactly
    one plan per process and no handle is needed.  Creating a second plan before
    destroying the first would overwrite that module state, hence the guard.
    """

    _live = False

    def __init__(self, dec, static_tdma=True):
        if PoissonPlan._live:
            raise RuntimeError("a PoissonPlan already exists in this process; "
                               "destroy it first (the Fortran plan is a singleton)")
        self.dec = dec

        _lib.poisson_plan_create(
            dec.c1.rank, dec.c2.rank, dec.c3.rank,
            dec.c1.size, dec.c2.size, dec.c3.size,
            dec.c1.west, dec.c2.west, dec.c3.west,
            dec.c1.east, dec.c2.east, dec.c3.east,
            dec.c1.handle, dec.c2.handle, dec.c3.handle,
            dec.n1, dec.n2, dec.n3,
            dec.n1sub, dec.n2sub, dec.n3sub,
            dec.L1, dec.L2, dec.L3,
            int(dec.pbc[0]), int(dec.pbc[1]), int(dec.pbc[2]),
            dec.threads_tdma[0], dec.threads_tdma[1], dec.threads_tdma[2],
            dec.threads_fft[0], dec.threads_fft[1], dec.threads_fft[2])
        PoissonPlan._live = True

        _lib.poisson_cudecomp_init(dec.n2msub_Isub, dec.h1psub_Jsub)

        if static_tdma:
            # Factor the invariant spectral operator once, outside the solve loop.
            _lib.poisson_tdma_static_init(
                dec.h1psub_Jsub, dec.n2m, dec.n3msub,
                ctypes.c_void_p(dec.dx3_d.data.ptr),
                ctypes.c_void_p(dec.dmx3_d.data.ptr),
                dec.h1psub_Jsub_ista)

    def solve(self, rhs, p):
        """Solve once; p receives the solution and rhs is overwritten.

        The solver reuses rhs as working storage, so a repeated-solve loop must
        rebuild the right-hand side before every call, as a CFD time step would.
        """
        if not PoissonPlan._live:
            raise RuntimeError("plan already destroyed")
        d = self.dec
        _lib.poisson_solve(
            _devptr(rhs, d.rhs_shape, "rhs"),
            _devptr(p, d.p_shape, "p"),
            ctypes.c_void_p(d.dmx1_d.data.ptr),
            ctypes.c_void_p(d.dmx2_d.data.ptr),
            ctypes.c_void_p(d.dmx3_d.data.ptr),
            ctypes.c_void_p(d.dx3_d.data.ptr),
            d.h1psub, d.h1psub_Jsub, d.n2msub_Isub, d.n1msub_Jsub,
            d.countsendI, d.countdistI, d.countsendJ, d.countdistJ,
            d.ddt_dble_C_in_C2I, d.ddt_dble_I_in_C2I,
            d.ddt_dble_J_in_C2J, d.ddt_dble_C_in_C2J,
            d.ddt_cplx_I_in_C2I, d.ddt_cplx_C_in_C2I,
            d.ddt_cplx_J_in_C2J, d.ddt_cplx_C_in_C2J,
            d.iend, d.ista, d.jend, d.jsta,
            d.h1psub_Jsub_ista, d.n2msub_Isub_jsta, d.n1msub_Jsub_ista)

    def timer_clear(self):
        """Drop the computation/communication accumulators.  Call after warm-up.

        No-op unless the environment has POISSON_TIMER=1.
        """
        _lib.poisson_timer_clear()

    def timer_read(self):
        """(computation, communication, total, nsolve) for this rank since the last clear.

        Seconds, summed over solves, not averaged.  The solver charges every MPI and NCCL call to
        communication and everything else to computation, synchronizing the device at each
        boundary; computation + communication equals total by construction, and the two are
        accumulated through separate stamp chains, so their ratio is a check that no part of the
        solve escaped instrumentation.  All zero when POISSON_TIMER is unset.
        """
        a, b, c = ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0)
        n = ctypes.c_int(0)
        _lib.poisson_timer_read(ctypes.byref(a), ctypes.byref(b),
                                ctypes.byref(c), ctypes.byref(n))
        return a.value, b.value, c.value, n.value

    def sync(self):
        """Block until the GPU is done; needed before timing or reading results back."""
        istat = ctypes.c_int(0)
        _lib.poisson_sync(ctypes.byref(istat))
        if istat.value != 0:
            raise RuntimeError(f"cudaDeviceSynchronize failed (status {istat.value})")

    def destroy(self):
        if PoissonPlan._live:
            _lib.poisson_destroy()
            PoissonPlan._live = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()
