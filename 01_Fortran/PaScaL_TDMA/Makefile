# Compiler and complilation flag are included in Makefile.inc
include Makefile.inc

lib:
	mkdir -p include;mkdir -p lib
	cd src; mkdir -p obj; $(MAKE) all

all:
	mkdir -p include; mkdir -p lib
	cd src; mkdir -p obj; $(MAKE) all

clean:
	cd src; $(MAKE) clean
