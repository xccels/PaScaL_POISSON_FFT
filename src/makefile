PROJECT_ROOT = ..
include $(PROJECT_ROOT)/makefile.inc
LIB = libfft_poisson.a

OBJDIR = ./obj
LIBDIR = ../lib
INCLUDEDIR = ../include

# 필요한 파일들
SRCFILES = nvtx.f90 para_range.f90 fft_3d_poisson.f90
OBJS = $(addprefix $(OBJDIR)/, $(SRCFILES:.f90=.o))

PaScaL_HOME = ../PaScaL_TDMA
PaScaLINC = -I$(PaScaL_HOME)/include
PaScaLLIB = -L$(PaScaL_HOME)/lib -lpascal_tdma

NVTXFLAGS = -L/usr/local/cuda/lib64 -lnvToolsExt -DUSE_NVTX
CUDA_FLAGS = -cuda -cudalib=cufft,cublas -Minfo=accel
ACC_FLAGS = -acc=gpu

PRECISION_FLAGS = -DDOUBLE_PRECISION
# PRECISION_FLAGS = -DSINGLE_PRECISION

# Compiler flags
FC_FLAGS = ${NVTXFLAGS} ${CUDA_FLAGS} ${ACC_FLAGS} ${PRECISION_FLAGS} ${PaScaLINC} ${PaScaLLIB}

# 디렉토리가 존재하지 않으면 생성
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(LIBDIR):
	mkdir -p $(LIBDIR)

# .f90 파일을 .o 파일로 컴파일
$(OBJDIR)/%.o: %.f90 | $(OBJDIR)
	$(FC) $(FLAG) -c $< -o $*.o -module $(OBJDIR) -I$(OBJDIR) $(FC_FLAGS)
	mv $*.o $(OBJDIR)/

# 기본 타겟
all: $(OBJDIR) $(LIBDIR) $(LIB)

# 정적 라이브러리 생성
$(LIB): $(OBJS)
	ar rcs $(LIBDIR)/$(LIB) $(OBJS)
	if [ ! -d $(INCLUDEDIR) ]; then mkdir -p $(INCLUDEDIR); fi
	cp $(OBJDIR)/*.mod $(INCLUDEDIR)/

# 생성된 파일들 정리
clean:
	rm -rf $(OBJDIR)
	rm -f $(LIBDIR)/*
	rm -f $(INCLUDEDIR)/*
	rm -rf lib
	rm -rf $(PROJECT_ROOT)/run/data

# 데이터 디렉토리 제거
rm:
	rm -rf $(PROJECT_ROOT)/run/data

# MPI로 실행
exe:
	mpirun $(LIBDIR)/$(LIB)

# NVTX로 프로파일링
nvtx:
	nsys profile --stats=true --trace=nvtx,cuda --cuda-memory-usage=true --force-overwrite true -o Output mpirun $(LIBDIR)/$(LIB)

.PHONY: all clean rm exe nvtx
