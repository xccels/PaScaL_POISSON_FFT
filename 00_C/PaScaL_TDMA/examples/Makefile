HOME = ../..

include $(HOME)/Makefile.inc

EXE = a.out
EXE    = a.out
OBJDIR = ./obj

INC = ../include
LIB = ../lib -lpascal_tdma
SRC = global.cpp \
      mpi_topology.cpp \
      mpi_subdomain.cpp \
      solve_theta.cpp \
      main.cpp

global = global.o
topology = mpi_topology.o 
subdomain = mpi_subdomain.o
THETA = solve_theta.o
main = main.o
OBJS = $(addprefix $(OBJDIR)/,$(SRC:.cpp=.o))

OBJS   = $(global) $(topology) $(subdomain) $(THETA) $(main)
INC = -I../src
LIBS = -L../lib -lpascal_tdma

all:$(OBJS)
	$(FC) ${FLAG} ${OBJS} -o $(EXE) -I$(INC) -L$(LIB)
	mv a.out ../run
$(OBJDIR)/%.o: %.cpp modules.hpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I. $(INC)

%.o: %.f90
	$(FC) $(FLAG) -c $^ -o $@  -I$(INC)
all: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXE) $(INC) $(LIBS)
	mv $(EXE) ../run

%(subdomain)	: %(topology)
%(THETA)		: %(global) #%(p3t_module) 
%(main)			: %(global) %(topology) %(subdomain) %(THETA)
$(EXE)			: %(OBJS)
$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -f *.mod *.o *__genmod.f90 ../run/$(EXE) ../run/*.PLT ../run/*.dat
	rm -rf $(OBJDIR) ../run/$(EXE)