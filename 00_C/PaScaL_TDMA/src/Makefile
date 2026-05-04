HOME = ../..
include $(HOME)/Makefile_TDMA.inc

LIB = libpascal_tdma.a
OBJDIR = ./obj
SRCS = para_range.cpp tdmas.cpp pascal_tdma.cpp timer.cpp
OBJS = $(SRCS:%.cpp=$(OBJDIR)/%.o)

RANGE = $(OBJDIR)/para_range.o
TDMAS = $(OBJDIR)/tdmas.o
PTDMA  = $(OBJDIR)/pascal_tdma.o
TIMER = $(OBJDIR)/timer.o

TDMAS_CUDA = $(OBJDIR)/tdmas_cuda.o
PTDMA_CUDA  = $(OBJDIR)/pascal_tdma_cuda.oa

OBJS   = $(RANGE) $(TIMER) $(TDMAS) $(PTDMA)

ifdef USE_CUDA
    OBJS += $(TDMAS_CUDA) $(PTDMA_CUDA)
endif
.PHONY: all inc clean

all: $(LIB) inc

$(LIB): $(OBJS)
	ar qc $@ $(OBJS)
	cp $@ ../lib

inc: $(LIB)
	cp $(OBJDIR)/*.mod ../include
$(LIB): $(OBJS) ; ar qc $@ $^ && cp $@ ../lib

$(OBJDIR)/%.o: %.f90
	$(FC) $(FLAG) -c $^ -o $@  -module $(OBJDIR) -I$(OBJDIR)
$(OBJDIR)/%.o: %.cuf
	$(FC) $(FLAG) -c $^ -o $@  -module $(OBJDIR) -I$(OBJDIR)
$(OBJDIR)/%.o:$(OBJDIR)/%.mod
inc: ; cp pascal_tdma.hpp ../include

%(PTDMA)	 : %(RANGE) %(TDMAS)

#$(OBJDIR)/tdmas.o:        CXXFLAGS += $(CXXFLAGS_REPORT)
# $(OBJDIR)/pascal_tdma.o:  CXXFLAGS += $(CXXFLAGS_REPORT)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR) ; $(CXX) $(CXXFLAGS) -I. -c $< -o $@ > $@.log 2>&1

clean:
	rm -rf ../lib ../include *__genmod.f90 $(LIB)
	rm -rf $(OBJDIR)
$(OBJDIR): ; mkdir -p $@

clean: ; rm -rf $(OBJDIR) $(LIB)