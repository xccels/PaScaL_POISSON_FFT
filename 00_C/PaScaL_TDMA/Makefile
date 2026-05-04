# Compiler and complilation flag are included in Makefile.inc
# Build PaScaL_TDMA library and example programs

include ./Makefile.inc

.PHONY: lib example all clean

lib:
	mkdir -p include;mkdir -p lib
	cd src; mkdir -p obj; make all
	mkdir -p include lib
	cd src && mkdir -p obj && $(MAKE) all

example:
	cd examples; make all
	$(MAKE) -C examples all

all:
	mkdir -p include; mkdir -p lib
	cd src; mkdir -p obj; make all
	cd examples; make all
all: lib example

clean:
	cd src; make clean
	cd examples; make clean
	$(MAKE) -C src clean
	$(MAKE) -C examples clean