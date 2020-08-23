
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "convolve.h"


void cube_setup(int c) {

	cubeSide = c;
	cubeNpts = cubeSide * cubeSide * cubeSide;
	blocksPerSide = cubeSide / B;
	blocksPerCube = cubeNpts / B_3;

}
