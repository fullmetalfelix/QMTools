
CC      = nvcc
CCFLAGS = -O3 -rdc=true --resource-usage -arch=sm_60 --compiler-options '-fPIC'
LFLAGS  = -lm

SRC = vectors.cu basisset.cu molecule.cu kernel-qube.cu kernel-hartree.cu automaton.cu qmtools.cu # randoms.cu kernel-rpts.cu kernel-acsf.cu kernel-qube.cu ga.cu cube.cu kernel.cu vectors.cu convolve.cu kernel-nn.cu

all:
	$(CC) $(CCFLAGS) --shared -o libqmtools.so $(SRC) $(LFLAGS) #-maxrregcount 32

ptx:
	$(CC) $(CCFLAGS) -c automaton.cu -maxrregcount 32

clean:
	rm -f *.o


