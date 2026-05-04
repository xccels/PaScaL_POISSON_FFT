# Usage:
#   make LANG=C [target]       - build C version
#   make LANG=Fortran [target] - build Fortran version (default)
#
# C targets      : all, lib, exe, clean, execlean
# Fortran targets: GPU_all, CPU_all, GPU_clean, CPU_clean

LANG ?= Fortran

ifeq ($(LANG),C)
    SUBDIR = 00_C
else ifeq ($(LANG),Fortran)
    SUBDIR = 01_Fortran
else
    $(error Unsupported LANG=$(LANG). Use LANG=C or LANG=Fortran)
endif

.PHONY: all lib exe clean execlean GPU_all CPU_all GPU_clean CPU_clean

all lib exe clean execlean GPU_all CPU_all GPU_clean CPU_clean:
	$(MAKE) -C $(SUBDIR) $@
