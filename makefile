# Compiler와 컴파일 플래그를 Makefile.inc에 포함
include ./makefile.inc

lib:
	mkdir -p include lib
	$(MAKE) -C PaScaL_TDMA lib || true
	$(MAKE) -C src all || true

example:
	mkdir -p examples/obj
	$(MAKE) -C examples all || true

all: lib example

exe:
	$(MAKE) -C examples exe

clean:
	$(MAKE) -C PaScaL_TDMA clean || true
	$(MAKE) -C src clean || true
	$(MAKE) -C examples clean || true
	rm -rf core.*
	rm -rf run/core.*

rm:
	$(MAKE) -C src rm || true
	$(MAKE) -C examples rm || true
	rm -rf core.*
	rm -rf run/core.*

.PHONY: lib example all clean rm
