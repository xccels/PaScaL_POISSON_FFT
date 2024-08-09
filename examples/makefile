PROJECT_ROOT = ..
include $(PROJECT_ROOT)/makefile.inc
EXE = examples.exe

OBJDIR = ./obj

OBJS = nvtx global mpi_topology interface mpi_subdomain cuda_subdomain para_range cuda_pressure cuda_post main

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC = -I$(PaScaL_HOME)/include
PaScaLLIB = -L$(PaScaL_HOME)/lib -lpascal_tdma

FFT_POI_HOME = ..
FFT_POIINC = -I$(FFT_POI_HOME)/include
FFT_POILIB = -L$(FFT_POI_HOME)/lib -lfft_poisson

SRC = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(OBJS)))

NVTXFLAGS = -L/usr/local/cuda/lib64 -lnvToolsExt -DUSE_NVTX
CUDA_FLAGS = -cuda -cudalib=cufft,cublas -Minfo=accel
ACC_FLAGS = -acc=gpu

PRECISION_FLAGS = -DDOUBLE_PRECISION
# PRECISION_FLAGS = -DSINGLE_PRECISION

# Compile each .f90 file into an object file
$(OBJDIR)/%.o: %.f90
	${FC} ${FLAG} -c $< -o $(OBJDIR)/$*.o -module $(OBJDIR) -I$(OBJDIR) ${NVTXFLAGS} ${CUDA_FLAGS} ${ACC_FLAGS} ${PRECISION_FLAGS} ${FFT_POIINC} ${PaScaLINC} 

# Link all object files into the final executable
all: $(SRC)
	mkdir -p obj
	${FC} ${FLAG} -o ${EXE} $(SRC) ${NVTXFLAGS} ${CUDA_FLAGS} ${ACC_FLAGS} ${PRECISION_FLAGS} ${FFT_POILIB} ${PaScaLLIB} -lcufft -lcublas
	mv $(EXE) ../run

clean:
	rm -rf $(OBJDIR)
	rm -f $(PROJECT_ROOT)/run/$(EXE)
	rm -rf $(PROJECT_ROOT)/run/data

rm:
	rm -rf $(PROJECT_ROOT)/run/data

exe:
	mpirun $(PROJECT_ROOT)/run/${EXE}

nvtx:
	nsys profile --stats=true --trace=nvtx,cuda --cuda-memory-usage=true  --force-overwrite true -o Output mpirun $(PROJECT_ROOT)/run/${EXE}
