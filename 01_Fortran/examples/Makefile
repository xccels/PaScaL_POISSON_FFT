EXE    = unified_poisson.exe
OBJDIR = ./obj
VPATH  = . ../src

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC   = -I$(PaScaL_HOME)/include
PaScaLLIB   = -L$(PaScaL_HOME)/lib -lpascal_tdma

ifdef USE_GPU
include ../Makefile_GPU.inc

OBJS_NAMES = nvtx global mpi_topology interface mpi_subdomain cuda_subdomain \
             para_range cuda_pressure cuda_post fft_3d_poisson main

NVTXFLAGS       = -L/usr/local/cuda/lib64 -lnvToolsExt -DUSE_NVTX
CUDA_FLAGS      = -cuda -cudalib=cufft,cublas -Minfo=accel
ACC_FLAGS       = -acc=gpu
PRECISION_FLAGS = -DDOUBLE_PRECISION
GPU_FLAGS       = -DUSE_GPU

COMPILE_FLAGS = $(NVTXFLAGS) $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) $(GPU_FLAGS) $(PaScaLINC)
LINK_FLAGS    = $(NVTXFLAGS) $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) $(GPU_FLAGS) \
                $(PaScaLLIB) -lcufft -lcublas

else
include ../Makefile_CPU.inc

MKL_HOME  = /opt/intel/oneapi/mkl/latest
FFTWINC   = -I$(MKL_HOME)/include/fftw
FFTWFLAGS = -mkl=sequential

OBJS_NAMES = global mpi_topology module_mpi_subdomain module_solve_poisson module_post main

COMPILE_FLAGS = $(FFTWINC) $(FFTWFLAGS) $(PaScaLINC) -cpp
LINK_FLAGS    = $(PaScaLLIB) $(FFTWFLAGS)
endif

OBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(OBJS_NAMES)))

$(OBJDIR)/%.o: %.f90
	$(FC) $(FLAG) -c $< -o $@ -module $(OBJDIR) -I$(OBJDIR) $(COMPILE_FLAGS)

all: $(OBJDIR) $(OBJS)
	$(FC) $(FLAG) -o $(EXE) $(OBJS) $(LINK_FLAGS)
	mv $(EXE) ../run

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR)
	rm -f ../run/$(EXE)

.PHONY: all clean
