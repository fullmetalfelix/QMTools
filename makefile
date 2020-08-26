
CC      = nvcc
CCFLAGS = -O3 -rdc=true --resource-usage -arch=sm_60
LFLAGS  = -lm -DDEBUGPRINT

SRC = cube.cu kernel.cu vectors.cu convolve.cu main.cu

all:
	$(CC) $(CCFLAGS) -o convolver.exe $(SRC) $(LFLAGS)

clean:
	rm -f *.o
