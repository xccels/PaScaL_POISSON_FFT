PROJECT_ROOT = ..
include $(PROJECT_ROOT)/makefile.inc

MODE ?= debug
BUILD_FLAGS ?=

LIB = libfft_poisson.a
OBJDIR = ./obj/$(MODE)
LIBDIR = ../lib/$(MODE)
INCLUDEDIR = ../include/$(MODE)

SRCFILES = para_range.f90 poisson_timer.f90 fft_cudecomp_bridge.f90 fft_3d_poisson.f90
OBJS = $(addprefix $(OBJDIR)/, $(SRCFILES:.f90=.o))

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC = -I$(PaScaL_HOME)/include/$(MODE)
PaScaLLIB = -L$(PaScaL_HOME)/lib/$(MODE) -lpascal_tdma

NVTXFLAGS = -L/usr/local/cuda/lib64
CUDA_FLAGS = -cuda -cudalib=cufft,cublas -Minfo=accel $(GPU_ARCH_FLAGS)
ACC_FLAGS = -acc=gpu
PRECISION_FLAGS = -DDOUBLE_PRECISION
CUDECOMP_HOME ?= ../_install/cudecomp
CUDECOMP_FLAGS = -I$(CUDECOMP_HOME)/include -DPOISSON_USE_CUDECOMP
FC_FLAGS = $(CUDA_FLAGS) $(ACC_FLAGS) $(PRECISION_FLAGS) \
           $(PaScaLINC) $(PaScaLLIB) $(NVTXFLAGS) $(CUDECOMP_FLAGS) \
           $(BUILD_FLAGS)

all: $(OBJDIR) $(LIBDIR) $(INCLUDEDIR) $(LIBDIR)/$(LIB)

$(OBJDIR):
	mkdir -p $@

$(LIBDIR):
	mkdir -p $@

$(INCLUDEDIR):
	mkdir -p $@

$(OBJDIR)/%.o: %.f90 | $(OBJDIR)
	$(FC) $(FLAG) -c $< -o $@ -module $(OBJDIR) -I$(OBJDIR) $(FC_FLAGS)

$(OBJDIR)/fft_cudecomp_bridge.o: $(OBJDIR)/poisson_timer.o
$(OBJDIR)/fft_3d_poisson.o: $(OBJDIR)/fft_cudecomp_bridge.o $(OBJDIR)/poisson_timer.o

$(LIBDIR)/$(LIB): $(OBJS) | $(LIBDIR) $(INCLUDEDIR)
	ar rcs $@ $(OBJS)
	cp $(OBJDIR)/*.mod $(INCLUDEDIR)/

clean:
	rm -rf ./obj/debug* ./obj/perf*
	rm -rf ../lib/debug* ../lib/perf*
	rm -rf ../include/debug* ../include/perf*
	rm -rf $(PROJECT_ROOT)/run/data

rm:
	rm -rf $(PROJECT_ROOT)/run/data

.PHONY: all clean rm
