# =============================================================================
# PaScaL_POISSON_FFT - Python C-API wrapper library
#
#   Target: lib/libpoisson_capi.so
#           Shared library bundling the bind(C) wrapper with the already built
#           static libraries of 01_Fortran, called from Python via ctypes with
#           CuPy device arrays and mpi4py handles.
#
#   The 01_Fortran build is reused as-is: makefile.inc already compiles every
#   library source with -fPIC, so the archives can be linked into a .so without
#   recompiling them.  Build them first:
#
#     source 01_Fortran/load_neuron_env.sh
#     make -C 01_Fortran perf-c2i-direct
#
#   Usage:
#     make                        # build lib/libpoisson_capi.so
#     make MODE=debug_c2i_direct  # link against the detailed-profile build
#     make clean
# =============================================================================
POISSON_HOME ?= $(CURDIR)/../01_Fortran

# Which 01_Fortran build to link against.  perf_* leaves the per-phase
# synchronisation out of the solver, which is what a Python driver wants;
# debug_* keeps the 21-boundary profiler for diagnosis.
MODE ?= perf_c2i_direct

include $(POISSON_HOME)/makefile.inc

# makefile.inc declares GPU_ARCH_FLAGS empty, so a plain '?=' here would never
# fire.  Test the value instead: an empty list makes nvfortran target only the
# GPU of the build host, and the code then dies on any other architecture.
ifeq ($(strip $(GPU_ARCH_FLAGS)),)
GPU_ARCH_FLAGS := -gpu=cc70,cc80,cuda12.9
endif

SRCDIR = ./src
OBJDIR = ./obj
LIBDIR = ./lib
TARGET = $(LIBDIR)/libpoisson_capi.so

OBJS = $(OBJDIR)/poisson_capi.o

CUDA_FLAGS      = -cuda -cudalib=cufft,cublas $(GPU_ARCH_FLAGS)
ACC_FLAGS       = -acc=gpu
PRECISION_FLAGS = -DDOUBLE_PRECISION

CUDECOMP_HOME  ?= $(POISSON_HOME)/_install/cudecomp
CUDECOMP_INC    = -I$(CUDECOMP_HOME)/include
# rpath is baked in so ctypes.CDLL finds cuDecomp without LD_LIBRARY_PATH.
CUDECOMP_LIB    = -L$(CUDECOMP_HOME)/lib \
                  -Wl,-rpath,$(abspath $(CUDECOMP_HOME)/lib) \
                  -lcudecomp_fort -lcudecomp
CUDECOMP_FLAGS  = $(CUDECOMP_INC) -DPOISSON_USE_CUDECOMP

POISSON_INC = -I$(POISSON_HOME)/include/$(MODE)
POISSON_LIB = -L$(POISSON_HOME)/lib/$(MODE) -lfft_poisson
TDMA_INC    = -I$(POISSON_HOME)/PaScaL_TDMA/include/$(MODE)
TDMA_LIB    = -L$(POISSON_HOME)/PaScaL_TDMA/lib/$(MODE) -lpascal_tdma

COMPILE_FLAGS = $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) \
                $(POISSON_INC) $(TDMA_INC) $(CUDECOMP_FLAGS)

CHECK_LIBS = $(POISSON_HOME)/lib/$(MODE)/libfft_poisson.a \
             $(POISSON_HOME)/PaScaL_TDMA/lib/$(MODE)/libpascal_tdma.a \
             $(CUDECOMP_HOME)/lib/libcudecomp_fort.so

all: check $(TARGET)

# Fail with an actionable message instead of a wall of undefined symbols.
check:
	@for f in $(CHECK_LIBS); do \
	    if [ ! -f "$$f" ]; then \
	        echo "[ERROR] missing: $$f"; \
	        echo "        build 01_Fortran first, e.g."; \
	        echo "          source ../01_Fortran/load_neuron_env.sh"; \
	        echo "          make -C ../01_Fortran perf-c2i-direct"; \
	        exit 1; \
	    fi; \
	done

$(TARGET): $(OBJS) | $(LIBDIR)
	$(FC) $(FLAG) -shared $(OBJS) -o $@ \
	      $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) \
	      $(POISSON_LIB) $(TDMA_LIB) $(CUDECOMP_LIB) \
	      -lcufft -lcublas

$(OBJDIR)/%.o: $(SRCDIR)/%.f90 | $(OBJDIR)
	$(FC) $(FLAG) -c $< -o $@ -module $(OBJDIR) -I$(OBJDIR) $(COMPILE_FLAGS)

$(OBJDIR) $(LIBDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR) *.mod *__genmod.f90

.PHONY: all check clean
