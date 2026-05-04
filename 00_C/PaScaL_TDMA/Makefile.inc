###############################################################################
# Compile setting with intel compiler and intel MPI.
# Modify flags according to the system and environment
###############################################################################
# MPI C++ compiler wrapper. Override by setting CXX in the environment.
CXX ?= mpicxx

# FC = mpiifort

# FLAG  = -O3 -fPIC -cpp
# FLAG += -g
# FLAG += -fcheck=all -warn-all 
# FLAG += -fp-model=precise 
# FLAG += -check bounds
# FLAG += -xMIC-AVX512

###############################################################################
# Compile setting with nvfortran and non-intel MPI.
# Modify flags according to the system and environment
###############################################################################

FC = mpifort

FLAG  = -Ofast -fPIC -cpp 
FLAG += -tp=skylake 

###############################################################################
# To use CUDA
# PaScaL_TDMA is compiled for CUDA also.
###############################################################################

USE_CUDA = yes
FLAG += -mp:multicore -Mcuda -Minfo=accel
#FLAG += -DUSE_NVTX
# Start with user-provided flags but strip unsupported ones (e.g. -fPI).
#CXXFLAGS := $(filter-out -fPI,$(CXXFLAGS))

# Optimisation flags for building the static library
CXXFLAGS = -Ofast -fPIC -std=c++17 -tp=icelake -Minfo=all -mp

# Uncomment for extra debugging
#CXXFLAGS += -g -Wall -Wextra