
FC = nvfortran
CC = nvcc
FCFLAGS = -shared -O3 -fPIC -gpu=ccall -cuda -cudalib=cufft
CCFLAGS = -shared -O3 -rdc=true --gpu-architecture=sm_60 --compiler-options '-fPIC' #--resource-usage 
LFLAGS  = -lm -lc -Mcudalib=cufft # -lcudart -lcudadevrt
TARGET = libqmtools.so
OBJS = $(addsuffix .o,$(basename $(SRC))) 
SRC = vectors.cu basisset.cu molecule.cu kernel-qube.cu kernel-hartree.cu automaton.cu qmtools.cu \
	potential.cuf ftypes.cuf
#randoms.cu kernel-rpts.cu kernel-acsf.cu kernel-qube.cu ga.cu cube.cu kernel.cu vectors.cu convolve.cu kernel-nn.cu	

#$^ references all the prerequisites
#$@ is the target and $< is the (first) prerequisite

$(TARGET) : $(OBJS) link.o
	$(FC) -Mcuda -shared -o $@ $^ $(LFLAGS)
%.o: %.cu
	$(CC) $(CCFLAGS) --device-c $< --output-file $@ 
link.o: vectors.o basisset.o molecule.o kernel-qube.o kernel-hartree.o automaton.o qmtools.o
	$(CC) $(CCFLAGS) --device-link $^ --output-file $@
%.o: %.cuf
	$(FC) -c -fPIC ftypes.cuf
	$(FC) -c $(FCFLAGS) -o $@ $^ 

ptx:
	$(CC) $(CCFLAGS) -c automaton.cu -maxrregcount 32

#clean doesn't currently automatically remove the *.o or *.mod files, '$ make clean' is required for cleaning
clean:
	rm -f *.o *.mod



