
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "cube.h"
#include "convolve.h"
#include "vectors.h"



/// Number of cubes.
__constant__ unsigned int c_nCubes;

/// Total number of points in a cube grid.
__constant__ unsigned int c_cubeNpts;


/// Number of points along one side of the cube.
__constant__ int c_cubeSide;


/// Total number of GPU blocks assigned to one cube.
__constant__ int c_blocksPerCube;

/// Number of GPU blocks long one side of the cube.
__constant__ int c_blocksPerSide;


/// Diffusion constant of the A0 field.
__constant__ number c_A0_diff;

/// A0 generation factor.
__constant__ number c_A0_gen;

/// A0 spatial loss factor
__constant__ number c_A0_loss[3];



/* FLEXIBLE CUBE SIZE IMPLEMENTATION */

__constant__ uint3 c_cubeSideF;
__constant__ uint3 c_blocksPerCubeF;
__constant__ uint3 c_blocksPerSideF;








/// Loads a reference cube from a file into the CPU memory.
///
///
void cube_load_reference(Cube *cube, const char *filename) {

	printf("loading reference cube: %s ...\n", filename);

	FILE *fbin = fopen(filename, "rb");

	// read the molecule specification
	int natm;
	fread(&natm, sizeof(int), 1, fbin);

	cube->molecule.natoms = natm;
	cube->molecule.Zs = (int*)malloc(sizeof(int) * natm);
	cube->molecule.coords = (float3*)malloc(sizeof(float3) * natm);
	cube->molecule.qtot = 0;

	for (int i=0; i<natm; ++i) {
		fread(&cube->molecule.Zs[i], sizeof(int), 1, fbin);
		fread(&cube->molecule.coords[i], sizeof(float3), 1, fbin);
		cube->molecule.qtot += cube->molecule.Zs[i];
	}
	
	// read the grid shape
	fread(&cube->gridSize.x, sizeof(unsigned int), 1, fbin);
	fread(&cube->gridSize.y, sizeof(unsigned int), 1, fbin);
	fread(&cube->gridSize.z, sizeof(unsigned int), 1, fbin);
	cube->npts = cube->gridSize.x * cube->gridSize.y * cube->gridSize.z;
	printf("grid shape: [%i %i %i]\n", cube->gridSize.x, cube->gridSize.y, cube->gridSize.z);


	cube->gpu_grid.x = cube->gridSize.x / 8;
	cube->gpu_grid.y = cube->gridSize.y / 8;
	cube->gpu_grid.z = cube->gridSize.z / 8;
	printf("GPU GRID: [%i %i %i]\n", cube->gpu_grid.x, cube->gpu_grid.y, cube->gpu_grid.z);


	// read the grid origin
	fread(&cube->grid0.x, sizeof(float), 1, fbin);
	fread(&cube->grid0.y, sizeof(float), 1, fbin);
	fread(&cube->grid0.z, sizeof(float), 1, fbin);
	printf("grid origin: %f %f %f \n", cube->grid0.x, cube->grid0.y, cube->grid0.z);

	// translate coordinates so that 0 is at the origin of the grid
	for (int i=0; i<natm; ++i) {
		cube->molecule.coords[i].x -= cube->grid0.x;
		cube->molecule.coords[i].y -= cube->grid0.y;
		cube->molecule.coords[i].z -= cube->grid0.z;
	}

	// read the density cube
	cube->Q = (number*)malloc(sizeof(number) * cube->npts);
	fread(cube->Q, sizeof(number), cube->npts, fbin);

	fclose(fbin);
	printf("cube read.\n");
}

void cube_load_reference_dummy(Cube *cube) {

	printf("loading reference dummy...\n");

	// read the molecule specification
	int natm = 1;
	cube->molecule.natoms = natm;
	cube->molecule.Zs = (int*)malloc(sizeof(int) * natm);
	cube->molecule.coords = (float3*)malloc(sizeof(float3) * natm);

	cube->molecule.Zs[0] = 2;
	cube->molecule.coords[0].x = 0;
	cube->molecule.coords[0].y = 0;
	cube->molecule.coords[0].z = 0;
	cube->molecule.qtot = 2;

	
	// read the grid shape
	cube->gridSize.x = 64;
	cube->gridSize.y = 64;
	cube->gridSize.z = 64;
	cube->npts = cube->gridSize.x * cube->gridSize.y * cube->gridSize.z;
	printf("grid shape: [%i %i %i]\n", cube->gridSize.x, cube->gridSize.y, cube->gridSize.z);

	cube->gpu_grid.x = cube->gridSize.x / 8;
	cube->gpu_grid.y = cube->gridSize.y / 8;
	cube->gpu_grid.z = cube->gridSize.z / 8;
	printf("GPU GRID: [%i %i %i]\n", cube->gpu_grid.x, cube->gpu_grid.y, cube->gpu_grid.z);


	// read the grid origin
	cube->grid0.x = -5.0f * ANG2BOR;
	cube->grid0.y = -5.0f * ANG2BOR;
	cube->grid0.z = -5.0f * ANG2BOR;
	printf("grid origin: %f %f %f \n", cube->grid0.x, cube->grid0.y, cube->grid0.z);

	// translate coordinates so that 0 is at the origin of the grid
	for (int i=0; i<natm; ++i) {
		cube->molecule.coords[i].x -= cube->grid0.x;
		cube->molecule.coords[i].y -= cube->grid0.y;
		cube->molecule.coords[i].z -= cube->grid0.z;
	}

	// read the density cube
	cube->Q = (number*)calloc(sizeof(number), cube->npts);
	printf("cube read.\n");
}


void cube_debug_print(Cube *ref, number *gpusrc, const char *filename) {

	number *dst = (number*)malloc(sizeof(number) * ref->npts);
	cudaMemcpy(dst, gpusrc, sizeof(number) * ref->npts, cudaMemcpyDeviceToHost);
	
	unsigned int gpub = 8;

	FILE *fbin = fopen(filename, "wb");
	fwrite(&ref->npts, sizeof(unsigned int), 1, fbin);
	fwrite(&ref->gridSize, sizeof(dim3), 1, fbin);
	fwrite(&gpub, sizeof(unsigned int), 1, fbin);
	fwrite(dst, sizeof(number), ref->npts, fbin);
	fclose(fbin);


	free(dst);
}













/*

void cube_setup(Cube *cube, int ncubes, int c) {

	cube->nCubes = ncubes;
	cube->side = c;
	cube->npts = c * c * c;
	cube->blocksPerSide = c / B;
	cube->blocksPerCube = cube->npts / B_3;
	cube->totalPoints = cube->npts * ncubes;

	printf("# of cubes: %i\n", cube->nCubes);
	printf("# of voxels in cube: %i\n", cube->npts);
	printf("# of blocks/side: %i\n", cube->blocksPerSide);
	printf("# of blocks/cube: %i\n", cube->blocksPerCube);
	printf("allocation size of one grid: %li\n", cube->totalPoints);


	cudaError_t cudaError;

	cudaError = cudaMemcpyToSymbol(c_nCubes, &cube->nCubes, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpyToSymbol(c_cubeSide, &cube->side, sizeof(int), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpyToSymbol(c_cubeNpts, &cube->npts, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpyToSymbol(c_blocksPerSide, &cube->blocksPerSide, sizeof(int), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpyToSymbol(c_blocksPerCube, &cube->blocksPerCube, sizeof(int), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cube->gpu_block = dim3(B,B,B);
	cube->gpu_grid  = dim3(cube->blocksPerSide, cube->blocksPerSide, cube->blocksPerSide);
}

void cube_allocate(Cube* cube) {

	unsigned long n = cube->totalPoints;
	cudaError_t cudaError;

	cube->Q = (number*)malloc(sizeof(number) * n);
	cube->A0  = (number*)malloc(sizeof(number) * n);
	cube->PmQ = (number*)malloc(sizeof(number) * n);


	cudaError = cudaMalloc((void**)&cube->d_Q, sizeof(number) * n);
	assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cube->d_P, sizeof(number) * n);
	assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cube->d_Qnew, sizeof(number) * n);
	assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cube->d_PmQ, sizeof(number) * n);
	assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cube->d_A0,  sizeof(number) * n);
	assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cube->d_A0new, sizeof(number) * n);
	assert(cudaError == cudaSuccess);

	printf("total allocation size (GPU): %li B\n", n * 6 * sizeof(number));
}

void cube_free(Cube *cube) {

	free(cube->A0);
	cudaFree(cube->d_A0); cudaFree(cube->d_A0new);

	cudaFree(cube->d_P);

	free(cube->PmQ);
	cudaFree(cube->d_PmQ);
	free(cube->Q);
	cudaFree(cube->d_Q); cudaFree(cube->d_Qnew);
}

void cube_params(Cube *cube) {

	cudaError_t cudaError;
	number x;

	x = (number)(0.2 / 6.0);
	cudaError = cudaMemcpyToSymbol(c_A0_diff, &x, sizeof(number), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	x = (number)(1.0);
	cudaError = cudaMemcpyToSymbol(c_A0_gen,  &x, sizeof(number), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	number loss[3] = {0.3f, 0.1f, 0.01f};
	cudaError = cudaMemcpyToSymbol(c_A0_loss, &loss, sizeof(number)*3, 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);
}




void cube_populate(Cube* cube) {

	printf("populating cubes...\n");
	
	// set A field to 0
	memset(cube->A0,  0, sizeof(number) * cube->totalPoints);
	cudaMemcpy(cube->d_A0,  cube->A0,  sizeof(number)*cube->totalPoints, cudaMemcpyHostToDevice);


	unsigned int idx = cube->side/2 + cube->side * cube->side / 2 + cube->side * cube->side * cube->side / 2;
	// place nuclear charges
	memset(cube->PmQ, 0, sizeof(number) * cube->totalPoints);
	//for(int i=0; i<cube->nCubes; i++) {
		//cube->PmQ[idx+1] = (number)1;
	//}
	cudaMemcpy(cube->d_P, cube->PmQ, sizeof(number)*cube->totalPoints, cudaMemcpyHostToDevice);


	// put some electrons in Q
	memset(cube->PmQ, 0, sizeof(number) * cube->totalPoints);
	//for(int i=0; i<cube->nCubes; i++)
		cube->PmQ[idx] = (number)1;
	cudaMemcpy(cube->d_Q, cube->PmQ, sizeof(number)*cube->totalPoints, cudaMemcpyHostToDevice);
	

	// then do PmQ
	memset(cube->PmQ, 0, sizeof(number) * cube->totalPoints);
	//cube->PmQ[idx+1] = (number)1;
	cube->PmQ[idx] = -(number)1;
	cudaMemcpy(cube->d_PmQ, cube->PmQ, sizeof(number)*cube->totalPoints, cudaMemcpyHostToDevice);

	/*
	// then set PmQ = nuclear - electron charge grid
	for(int i=0; i<cube->nCubes; i++) {
		cube->PmQ[0 + i*cube->npts] = -(number)1;
		cube->PmQ[10 + i*cube->npts] = (number)1;
	}
	cudaMemcpy(cube->d_PmQ, cube->PmQ, sizeof(number)*cube->totalPoints, cudaMemcpyHostToDevice);
	* /
	
	printf("done.\n");
}

*/



