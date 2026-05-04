ifdef USE_GPU

FC    = mpif90
FLAG  = -O3 -Mpreprocess -fPIC -fortranlibs -Kieee -Mlarge_arrays
FLAG += -Mcuda -Minfo=accel
USE_CUDA = yes

else

FC   = mpiifort
FLAG = -O3 -fPIC -fp-model=precise -xCORE-AVX2 -assume byterecl -cpp

endif
