GPU_all:
	mkdir -p run
	$(MAKE) -C PaScaL_TDMA USE_GPU=1
	$(MAKE) -C examples   USE_GPU=1

CPU_all:
	mkdir -p run
	$(MAKE) -C PaScaL_TDMA
	$(MAKE) -C examples

GPU_clean:
	$(MAKE) -C PaScaL_TDMA clean USE_GPU=1
	$(MAKE) -C examples   clean USE_GPU=1

CPU_clean:
	$(MAKE) -C PaScaL_TDMA clean
	$(MAKE) -C examples   clean

.PHONY: GPU_all CPU_all clean_GPU clean_CPU
