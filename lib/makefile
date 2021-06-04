
FC = nvfortran
CC = nvcc
FCFLAGS = -shared -fPIC -Mcudalib=cufft
CCFLAGS = -O3 -rdc=true -arch=sm_60 -shared --compiler-options '-fPIC' #--resource-usage
LFLAGS  = -lm -lc -Mcudalib=cufft -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
#-arch=sm_60 -lm -rdc=true -lgfortran -lcufft 
#location for -lcufft: -I/usr/local/cuda/include -L/usr/local/cuda/lib64
TARGET = libqmtools.so
OBJS = vectors_link.o basisset_link.o molecule_link.o kernel-qube_link.o kernel-hartree_link.o \
	automaton_link.o qmtools_link.o fortran_test.o
# $(addsuffix .o,$(basename $(SRC)))
SRC = vectors.cu basisset.cu molecule.cu kernel-qube.cu kernel-hartree.cu automaton.cu qmtools.cu \
	fortran_test.cuf
#randoms.cu kernel-rpts.cu kernel-acsf.cu kernel-qube.cu ga.cu cube.cu kernel.cu vectors.cu convolve.cu kernel-nn.cu	

#$^ references all the prerequisites
#$@ is the target and $< is the (first) prerequisite

$(TARGET) : $(OBJS)
	$(FC) -Mcuda -shared -o $@ $^ $(LFLAGS)
#$(CC) -rdc=true --shared -o $@ $^ $(LFLAGS)
%_link.o: %.cu
	$(CC) $(CCFLAGS) --device-c --output-file $@ $<
	$(CC) $(CCFLAGS) --device-link --output-file $(addsuffix _link.o,$(basename $@)) $@ -lcudadevrt -lcudart
%.o: %.cuf
	$(FC) -c $(FCFLAGS) -o $@ $<
ptx:
	$(CC) $(CCFLAGS) -c automaton.cu -maxrregcount 32
#clean doesn't currently automatically remove the *.o files, '$ make clean' is required for cleaning
clean:
	rm -f *.o *.mod


