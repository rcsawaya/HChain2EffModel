include ../this_dir.mk
include ../options.mk

#Define Flags ----------

TENSOR_HEADERS=$(PREFIX)/itensor/all.h $(PREFIX)/itensor/mps/idmrg.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS)
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS)
LIBFLAGS=-L$(ITENSOR_LIBDIR) $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L$(ITENSOR_LIBDIR) $(ITENSOR_LIBGFLAGS)

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: EffHubbardBFGS

debug: EffHubbardBFGS-g

EffHubbardBFGS: nrutil.o EffHubbardBFGS_tijVij.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) nrutil.o EffHubbardBFGS_tijVij.o -o EffHubbardBFGS_tijVij $(LIBFLAGS)

EffHubbardBFGS-g: mkdebugdir .debug_objs/nrutil.o .debug_objs/EffHubbardBFGS_tijVij.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/nrutil.o .debug_objs/EffHubbardBFGS_tijVij.o -o EffHubbardBFGS_tijVij-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	@rm -fr *.o .debug_objs EffHubbardBFGS EffHubbardBFGS-g
