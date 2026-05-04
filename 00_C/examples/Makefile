HOME = ..
include $(HOME)/Makefile.inc

EXE = core.exe 

OBJDIR = ./obj

global      = $(OBJDIR)/module_global.o
module_mpi  = $(OBJDIR)/module_mpi_topology.o $(OBJDIR)/module_mpi_subdomain.o
solver      = $(OBJDIR)/module_solve_poisson.o
analysis    = $(OBJDIR)/module_post.o
timer       = $(OBJDIR)/timer.o
main        = $(OBJDIR)/main.o

OBJS        = $(global) $(module_mpi) $(solver) $(analysis) $(timer) $(main)

MKL_HOME    = /opt/intel/oneapi/mkl/2023.2.0
FFTWINC  = -I$(MKL_HOME)/include/fftw
FFTWLIB  = -L$(MKL_HOME)/lib/intel64 -lmkl_rt -lpthread -lm -ldl

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC = -I$(PaScaL_HOME)/include
PaScaLLIB = -L$(PaScaL_HOME)/lib -lpascal_tdma


# FFTWFLAGS 	=  -qmkl=sequential

$(OBJDIR)/%.o: %.cpp modules.hpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I. $(FFTWINC) $(PaScaLINC) $(FFTWFLAGS) -c $< -o $@ > $@.log 2>&1

all: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXE) $(PaScaLLIB) $(FFTWLIB) $(FFTWFLAGS)
	mv $(EXE) ../run

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR); rm -f ../run/$(EXE); rm -rf ../run/data

