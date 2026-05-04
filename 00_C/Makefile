include ./Makefile.inc

.PHONY: all lib exe clean execlean

all: lib exe

lib:
	mkdir -p run
	$(MAKE) -C PaScaL_TDMA lib

exe:
	mkdir -p examples/obj
	$(MAKE) -C examples all

clean:
	$(MAKE) -C PaScaL_TDMA clean
	$(MAKE) -C examples   clean

execlean:
	$(MAKE) -C examples clean
