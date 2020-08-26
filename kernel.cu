
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "cube.h"
#include "convolve.h"


#define transferRate 0.9f
#define diffusionRate 0.2f


__constant__ number c_parameters[DNASIZE];
__device__ unsigned int blockCount = 0;



//__global__ void gpu_cube_unwrap(number *cubein, number *cubeout);
//__global__ void gpu_A0_propagate(number *PmQ, number *A0, number *A0out);


/* *** CUBE WRAPPER *** ***************************************************** */

__global__ void gpu_cube_wrap(number *Q, number *Qout) {

	uint ridx;
	uint widx;

	ridx = threadIdx.x + blockIdx.x*B;
	ridx+= (threadIdx.y + blockIdx.y*B)*gridDim.x*B;
	ridx+= (threadIdx.z + blockIdx.z*B)*gridDim.x*B*gridDim.y*B;

	widx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	widx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[widx] = Q[ridx];
}

void cpu_cube_loadref(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;
	
	// cube->Q holds the unwrapped tensor
	// copy tensor to cnv.Qn
	cudaMemcpy(cnv->d_Qn, cube->Q, sizeof(number) * cnv->maxpts, cudaMemcpyHostToDevice);

	// wrap it and store in Q tensor
	gpu_cube_wrap<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Qn, cnv->d_Q);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_cube_wrap error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();

	// copy the Q tensor on GPU into the cube->Q space
	cudaError = cudaMemcpy(cube->Q, cnv->d_Q, sizeof(number) * cnv->maxpts, cudaMemcpyDeviceToHost);
	assert(cudaError == cudaSuccess);

}


/* ************************************************************************** */
/* *** CONV RESETTERS *** *************************************************** */


__global__ void gpu_convolver_reset(number *A0, number *P, number *Q, number *PmQ) {

	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	A0[gidx] = 0;
	P[gidx] = 0;
	Q[gidx] = 0;
	PmQ[gidx] = 0;
}

void convolver_reset(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	gpu_convolver_reset<<<cube->gpu_grid, cnv->gpu_block>>>(
		cnv->d_A0,
		cnv->d_P,
		cnv->d_Q,
		cnv->d_PmQ
	);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("RESET: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	cudaDeviceSynchronize();
}

/* ************************************************************************** */
/* *** PROTON SPREADER *** ************************************************** */

__global__ void gpu_convolver_makeP(
	float gridStep, 	// grid step size
	uint3 gridBlocks, 	// number of blocks for the GPU grid
	number *P, 
	number *Q, 
	number *PmQ, 
	float3 *coords, 
	int *Zs) {

	__shared__ float3 atom;
	__shared__ int Z;


	uint gidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// get the atom coords from global memory
	if(gidx == 0) {
		atom = coords[blockIdx.x];
		Z = Zs[blockIdx.x];
		//printf("makeP gridBlocks: %i %i %i \n", gridBlocks.x, gridBlocks.y, gridBlocks.z);
	}
	__syncthreads();


	// compute in which block the atom is located
	// it might be at the edge between two blocks

	// this is the index in the unwrapped grid where the thread
	// should place its share of the nuclear charge
	uint3 i0;
	i0.x = (uint)floorf(atom.x / gridStep) + threadIdx.x;
	i0.y = (uint)floorf(atom.y / gridStep) + threadIdx.y;
	i0.z = (uint)floorf(atom.z / gridStep) + threadIdx.z;

	// compute the block-wrapped coordinates of i0
	uint3 b0;
	b0.x = i0.x / B;
	b0.y = i0.y / B;
	b0.z = i0.z / B;

	// compute hte index in the block
	uint3 t0;
	t0.x = i0.x - b0.x * B;
	t0.y = i0.y - b0.y * B;
	t0.z = i0.z - b0.z * B;

	// final total index
	gidx = (b0.x + b0.y * gridBlocks.x + b0.z * gridBlocks.x * gridBlocks.y) * B_3;
	gidx += t0.x + t0.y * B + t0.z * B_2;


	/*if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
		printf("atom %i [%f %f %f] -> B[%i %i %i]T[%i %i %i]\n", blockIdx.x, atom.x, atom.y, atom.z,
			b0.x, b0.y, b0.z, t0.x, t0.y, t0.z);
	}*/


	// calculate how the charge factor
	atom.x = fabsf(atom.x - i0.x * gridStep) / gridStep;
	atom.y = fabsf(atom.y - i0.y * gridStep) / gridStep;
	atom.z = fabsf(atom.z - i0.z * gridStep) / gridStep;
	number p = atom.x * atom.y * atom.z * Z;

	// save the nuclear charge
	P[gidx] = p;
	PmQ[gidx] = p;

	#ifndef DEBUG
	t0.x = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	if(t0.x == 0) {
		//printf("writing charge: %f\n", (number)Z);
		Q[gidx] = (number)Z;
		PmQ[gidx] = p - (number)Z;
	}
	#endif
}

void convolver_makeP(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	// the kernel needs to know the position of the atoms
	// relative to the origin of their grid

	cudaError = cudaMemcpy(cnv->d_coords, cube->molecule.coords, sizeof(float3)*cube->molecule.natoms, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpy(cnv->d_Zs, cube->molecule.Zs, sizeof(int)*cube->molecule.natoms, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);


	// the block is a 2 cube
	dim3 block(2,2,2);
	// the grid has one block for each atom in the molecule
	dim3 grid(cube->molecule.natoms, 1, 1);

	// since the grid step is small (0.1ang) we can be quite sure
	// that two atoms will not be spread over the same grid points
	// because the molecules are optimised organics!

	gpu_convolver_makeP<<<grid, block>>>(
		DS,
		cube->gpu_grid,
		cnv->d_P,
		cnv->d_Q,
		cnv->d_PmQ,
		cnv->d_coords,
		cnv->d_Zs
	);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);

	cudaDeviceSynchronize();
}

/* ************************************************************************** */
/* *** FIELD PROPAGATOR *** ************************************************* */

__global__ void gpu_A0_propagate(number *PmQ, number *A0, number *A0out, uint *deltaQ) {

	__shared__ number buffer[Bp1_3];

	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// this is the direction 
	int dx, dy, dz, sidx;
	
	// data destination address in shared memory - for main block of data
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	
	// load the main block of data in shared mem
	buffer[sidx] = A0[gidx];
	__syncthreads();

	uint3 mapping;

	// compute directions for block search - block size x must be at least 6!!!
	dx = (threadIdx.x == 0) - (threadIdx.x == 1);
	dy = (threadIdx.x == 2) - (threadIdx.x == 3);
	dz = (threadIdx.x == 4) - (threadIdx.x == 5);

	// and the mapping
	mapping.x = threadIdx.y;
	mapping.y = ((dx != 0)*threadIdx.y + (dz != 0)*threadIdx.z);
	mapping.z = threadIdx.z;

	// TODO: is there a way to remove the branchings completely?
	if(threadIdx.x >= 6) { // this is for loading the edges
		
		dx = (threadIdx.z + (threadIdx.x & 1)*4 > 3) * (2*(threadIdx.z & 1)-1);
		dy = (threadIdx.z >> (threadIdx.x & 1));
		dy = (2*(dy & 1)-1) * (threadIdx.z < 4);
		dz = (threadIdx.x == 6) * (2 * (((threadIdx.z>>1) & 1)>0)-1);

		mapping.y = threadIdx.y;
		mapping.z = threadIdx.y;
	}

	// for corners
	if(threadIdx.x == 7 && threadIdx.z == 4) {

		mapping.y = (threadIdx.y & 1);
		mapping.x = ((threadIdx.y >> 1) & 1);
		mapping.z = ((threadIdx.y >> 2) & 1);
		
		dx = 2*(mapping.x == 1)-1;
		dy = 2*(mapping.y == 1)-1;
		dz = 2*(mapping.z == 1)-1;

		// this may not be necessary?!
		//mapping.y = mapping.x*Bp + mapping.y*Bp*Bp1 + mapping.z*Bp*Bp1_2;
		//mapping.z = 10; // flag for corners!
	}
	
	/*
		tx 		77777777
		ty		01234567 (any are actually ok)
		tz		44444444
		dx		--++--++ => oddQ(ty/2)
		dy 		-+-+-+-+ => oddQ(ty)
		dz 		----++++ => oddQ(ty/4)
		
		dx		dy 		dz 		destination 	source
		-1		-1		-1		0	0	0
		-1		+1		-1
		+1		-1		-1
		+1		+1		-1
		-1		-1		+1
		-1		+1		+1
		+1		-1		+1
		+1		+1		+1		Bp 	Bp 	Bp

	*/

	// compute global data offset
	gidx = 0;

	// first do the block offset - this should be ok for surfs and edges or corners alike!
	sidx = blockIdx.x + dx;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx;

	sidx = blockIdx.y + dy;
	sidx += ((sidx < 0) - (sidx == gridDim.y))*gridDim.y;
	gidx += sidx * gridDim.x;

	sidx = blockIdx.z + dz;
	sidx += ((sidx < 0) - (sidx == gridDim.z))*gridDim.z;
	gidx += sidx * gridDim.x * gridDim.y;
	gidx *= B_3;


	// now do the thread offset - the mapping is different for surf/edge/corner

	// these are dependent only on deltablock and work for all surf/edge/corner
	gidx += (dx==-1) * (B-1);
	gidx += (dy==-1) * (B-1)*B;
	gidx += (dz==-1) * (B-1)*B_2;

	// these work for all too - this will be all false for CORNERS though
	gidx += (dx == 0) * mapping.x;
	gidx += (dy == 0) * mapping.y * B;
	gidx += (dz == 0) * mapping.z * B_2;

	// this is the special one for corners
	//gidx += (mapping.z == 10) * mapping.x;


	// compute destination in shared memory
	// it is good for all!
	sidx = 0;
	sidx += ((dx == 0)*(mapping.x+1) + (dx == 1)*Bp);
	sidx += ((dy == 0)*(mapping.y+1) + (dy == 1)*Bp) * Bp1;
	sidx += ((dz == 0)*(mapping.z+1) + (dz == 1)*Bp) * Bp1_2;


	// the only thread that will work are (0:7,:,:) and (7,:,0:5)
	//mapping.x = 1 - (((threadIdx.x==7) * threadIdx.z) > 3);
	mapping.x = 1 - (((threadIdx.x==7) * threadIdx.z) > 4);

	// load the periphery - this loads the extra sides of the cube
	if(mapping.x == 1)
		buffer[sidx] = A0[gidx];
	__syncthreads();

	/*
		// TODO: load edges of the cube and corners to convolve more
		// tID.x 6 7 (all y,z) are not working...
		//.                         (z>>1 & 1)  z&1
		// (6,:,0) -> edge (:,-,-) ..... 0 ..... 
		// (6,:,1) -> edge (:,+,-) ..... 0 ..... 
		// (6,:,2) -> edge (:,-,+) ..... 1 ..... 
		// (6,:,3) -> edge (:,+,+) ..... 1 ..... 
		// (6,:,4) -> edge (-,:,-) ..... 0 ..... 0
		// (6,:,5) -> edge (+,:,-) ..... 0 ..... 1
		// (6,:,6) -> edge (-,:,+) ..... 1 ..... 0
		// (6,:,7) -> edge (+,:,+) ..... 1 ..... 1
		// (7,:,0) -> edge (-,-,:) ..... 0 ..... 0
		// (7,:,1) -> edge (+,-,:) ..... 0 ..... 1
		// (7,:,2) -> edge (-,+,:) ..... 1 ..... 0
		// (7,:,3) -> edge (+,+,:) ..... 1 ..... 1
		// direction of the search?
		// tx  666666667777
		// Otx 000000001111 tx & 1
		// Etx 111111110000 (tx & 1)==0
		// tz  012345670123
		// Xtz 012345674567
		// %>3 000011111111
		// dx  0000-+-+-+-+
		// dy  -+-+0000--++
		// dz  --++--++0000

		// msk  111100001111 => tz<4
		// tz   012345670123
		// XXX  012345670011
		// dy   -+-+0000--++
	*/

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	//printf("c_blocksPerSide %i \n",c_blocksPerSide);
	

	// propagate the field:
	number f0 = buffer[sidx];
	number Adiff = 0;

	#pragma unroll
	for(dx=-1; dx<=1; dx+=2) {

		gidx = sidx + dx;
		Adiff += buffer[gidx];

		gidx = sidx + dx*Bp1;
		Adiff += buffer[gidx];

		gidx = sidx + dx*Bp1_2;
		Adiff += buffer[gidx];

		#pragma unroll
		for(dy=-1; dy<=1; dy+=2) {

			gidx = sidx + dx + dy*Bp1;
			Adiff += buffer[gidx] * OneOverSqrt2;

			gidx = sidx + dx + dy*Bp1_2;
			Adiff += buffer[gidx] * OneOverSqrt2;

			gidx = sidx + dx*Bp1 + dy*Bp1_2;
			Adiff += buffer[gidx] * OneOverSqrt2;

			#pragma unroll
			for(dz=-1; dz<=1; dz+=2) {
				gidx = sidx + dx + dy*Bp1 + dz*Bp1_2;
				Adiff += buffer[gidx] * OneOverSqrt3;
			}
		}

	}

	// Adiff = Sum_i c_i q_i for i in the neighbours
	// Adiff => Sum_i c_i q_i / Sum_i c_i for i in the neighbours
	Adiff = Adiff * OneOverDIFFTOT - f0;
	Adiff *= c_parameters[PARAM_A0_DIFF];


	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// compute the final value of the filed in the voxel
	number field = f0 + Adiff; // adds the diffusion
	field += PmQ[gidx] * c_parameters[PARAM_A0_AGEN]; // add the field generation/adsorption from charges

	// whatever is left can be lost to the bath
	f0 = fabsf(field);
	number loss;
	loss  = exp(-c_parameters[PARAM_A0_LOS1] * f0);
	loss += exp(-c_parameters[PARAM_A0_LOS2] * f0*f0);
	loss += exp(-c_parameters[PARAM_A0_LOS3] * f0*f0*f0);
	loss = 1.0f - 0.334f*loss; // divide by 3
	field -= loss*field;
	
	// save the output
	A0out[gidx] = field;
	if(gidx == 0)
		deltaQ[0] = 0;
}

void cpu_A0_propagate(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	gpu_A0_propagate<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_PmQ, cnv->d_A0, cnv->d_A0n, cnv->d_deltaQmax);
	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	//printf("A error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	number *tmp = cnv->d_A0;
	cnv->d_A0 = cnv->d_A0n;
	cnv->d_A0n = tmp;

	
	//printf("A0 propagated.\n");
}

/* ************************************************************************** */
/* *** CHARGE PROPAGATOR *** ************************************************ */

__global__ void gpu_Q_propagate(number *Q, number *Qout, number *P, number *PmQ, number *A0, uint *deltaQ) {

	__shared__ number shQ[Bp1_3];
	__shared__ number shA[Bp1_3];


	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// this is the direction 
	int dx, dy, dz, sidx;
	
	// data destination address in shared memory - for main block of data
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	
	// load the main block of data in shared mem
	shQ[sidx] = Q[gidx];
	shA[sidx] = A0[gidx];
	__syncthreads();

	uint3 mapping;

	// compute directions for block search - block size x must be at least 6!!!
	dx = (threadIdx.x == 0) - (threadIdx.x == 1);
	dy = (threadIdx.x == 2) - (threadIdx.x == 3);
	dz = (threadIdx.x == 4) - (threadIdx.x == 5);

	// and the mapping
	mapping.x = threadIdx.y;
	mapping.y = ((dx != 0)*threadIdx.y + (dz != 0)*threadIdx.z);
	mapping.z = threadIdx.z;

	// TODO: is there a way to remove the branchings completely?
	if(threadIdx.x >= 6) { // this is for loading the edges
		
		dx = (threadIdx.z + (threadIdx.x & 1)*4 > 3) * (2*(threadIdx.z & 1)-1);
		dy = (threadIdx.z >> (threadIdx.x & 1));
		dy = (2*(dy & 1)-1) * (threadIdx.z < 4);
		dz = (threadIdx.x == 6) * (2 * (((threadIdx.z>>1) & 1)>0)-1);

		mapping.y = threadIdx.y;
		mapping.z = threadIdx.y;
	}

	// for corners
	if(threadIdx.x == 7 && threadIdx.z == 4) {

		mapping.y = (threadIdx.y & 1);
		mapping.x = ((threadIdx.y >> 1) & 1);
		mapping.z = ((threadIdx.y >> 2) & 1);
		
		dx = 2*(mapping.x == 1)-1;
		dy = 2*(mapping.y == 1)-1;
		dz = 2*(mapping.z == 1)-1;
	}
	
	// compute global data offset
	gidx = 0;

	// first do the block offset - this should be ok for surfs and edges or corners alike!
	sidx = blockIdx.x + dx;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx;

	sidx = blockIdx.y + dy;
	sidx += ((sidx < 0) - (sidx == gridDim.y))*gridDim.y;
	gidx += sidx * gridDim.x;

	sidx = blockIdx.z + dz;
	sidx += ((sidx < 0) - (sidx == gridDim.z))*gridDim.z;
	gidx += sidx * gridDim.x * gridDim.y;
	gidx *= B_3;


	// now do the thread offset - the mapping is different for surf/edge/corner

	// these are dependent only on deltablock and work for all surf/edge/corner
	gidx += (dx==-1) * (B-1);
	gidx += (dy==-1) * (B-1)*B;
	gidx += (dz==-1) * (B-1)*B_2;

	// these work for all too - this will be all false for CORNERS though
	gidx += (dx == 0) * mapping.x;
	gidx += (dy == 0) * mapping.y * B;
	gidx += (dz == 0) * mapping.z * B_2;


	// compute destination in shared memory
	// it is good for all!
	sidx = 0;
	sidx += ((dx == 0)*(mapping.x+1) + (dx == 1)*Bp);
	sidx += ((dy == 0)*(mapping.y+1) + (dy == 1)*Bp) * Bp1;
	sidx += ((dz == 0)*(mapping.z+1) + (dz == 1)*Bp) * Bp1_2;


	// the only thread that will work are (0:7,:,:) and (7,:,0:5)
	mapping.x = 1 - (((threadIdx.x==7) * threadIdx.z) > 4);

	// load the periphery - this loads the extra sides of the cube
	if(mapping.x == 1) {
		shA[sidx] = A0[gidx];
		shQ[sidx] = Q[gidx];
	}
	__syncthreads();

	/*
		// TODO: load edges of the cube and corners to convolve more
		// tID.x 6 7 (all y,z) are not working...
		//.                         (z>>1 & 1)  z&1
		// (6,:,0) -> edge (:,-,-) ..... 0 ..... 
		// (6,:,1) -> edge (:,+,-) ..... 0 ..... 
		// (6,:,2) -> edge (:,-,+) ..... 1 ..... 
		// (6,:,3) -> edge (:,+,+) ..... 1 ..... 
		// (6,:,4) -> edge (-,:,-) ..... 0 ..... 0
		// (6,:,5) -> edge (+,:,-) ..... 0 ..... 1
		// (6,:,6) -> edge (-,:,+) ..... 1 ..... 0
		// (6,:,7) -> edge (+,:,+) ..... 1 ..... 1
		// (7,:,0) -> edge (-,-,:) ..... 0 ..... 0
		// (7,:,1) -> edge (+,-,:) ..... 0 ..... 1
		// (7,:,2) -> edge (-,+,:) ..... 1 ..... 0
		// (7,:,3) -> edge (+,+,:) ..... 1 ..... 1
		// direction of the search?
		// tx  666666667777
		// Otx 000000001111 tx & 1
		// Etx 111111110000 (tx & 1)==0
		// tz  012345670123
		// Xtz 012345674567
		// %>3 000011111111
		// dx  0000-+-+-+-+
		// dy  -+-+0000--++
		// dz  --++--++0000

		// msk  111100001111 => tz<4
		// tz   012345670123
		// XXX  012345670011
		// dy   -+-+0000--++
	*/

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	//printf("c_blocksPerSide %i \n",c_blocksPerSide);

	// trasfer charges based on the field
	number t, qfinal = 0;
	number q1 = shQ[sidx];
	number field1 = shA[sidx];
	number qdiff = 0, qtrans = 0;

	// transfer along the xy plane, diagonal
	#pragma unroll
	for(dx=-1; dx<=1; dx+=2) {

		// do the X axis
		gidx = sidx + dx;
		t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1);
		t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
		qtrans -= t;
		qdiff += shQ[gidx];
		//if(sidx == 1 + Bp1 + Bp1_2)
		//	printf("%i %i %i %i %i %i to dx %i ==> %e\n",blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z,dx,t);

		gidx = sidx + dx*Bp1;
		t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1);
		t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
		qtrans -= t;
		qdiff += shQ[gidx];

		gidx = sidx + dx*Bp1_2;
		t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1);
		t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
		qtrans -= t;
		qdiff += shQ[gidx];

		#pragma unroll
		for(dy=-1; dy<=1; dy+=2) {

			// do the xy plane
			gidx = sidx + dx + dy*Bp1;
			t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1) * OneOverSqrt2;
			t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
			qtrans -= t;
			qdiff += shQ[gidx] * OneOverSqrt2; // diffusion term

			// do the xz plane
			gidx = sidx + dx + dy*Bp1_2;
			t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1) * OneOverSqrt2;
			t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
			qtrans -= t;
			qdiff += shQ[gidx] * OneOverSqrt2; // diffusion term

			// do the yz plane
			gidx = sidx + dx*Bp1 + dy*Bp1_2;
			t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1) * OneOverSqrt2;
			t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
			qtrans -= t;
			qdiff += shQ[gidx] * OneOverSqrt2; // diffusion term


			#pragma unroll
			for(dz=-1; dz<=1; dz+=2) {

				gidx = sidx + dx + dy*Bp1 + dz*Bp1_2;
				t = tanhf(q1 * shA[gidx] - shQ[gidx] * field1) * OneOverSqrt3;
				t *= (t > 0) * q1 + (t <= 0) * shQ[gidx];
				qtrans -= t;
				qdiff += shQ[gidx] * OneOverSqrt3;
			}

		}
	}
	
	qtrans = qtrans * OneOverDIFFTOT;
	qtrans *= c_parameters[PARAM_QQ_TRNS];
	
	qdiff = qdiff * OneOverDIFFTOT - q1;
	qdiff *= c_parameters[PARAM_QQ_DIFF];

	qfinal = q1 + qtrans + qdiff;
	// BUG? the transfer causes loss of electrons?

	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[gidx] = qfinal;
	PmQ[gidx] = P[gidx] - qfinal;


	// compute the change in charge inside this block
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
	qdiff = fabsf(q1 - qfinal);
	sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2; // reindexing of shared mem
	shQ[sidx] = qdiff;
	__syncthreads();

	// sum them all up
	for(unsigned short stride=1; stride < B_3; stride*=2) {

		dx = sidx + stride;
		dy = dx * (dx < B_3);
		qdiff += shQ[dy] * (dx < B_3);
		__syncthreads();

		shQ[sidx] = qdiff;
		__syncthreads();
	}


	// now we have a partial sum - first thread writes the partial
	if(sidx == 0) {

		mapping.x = ((uint*)(&qdiff))[0];
		mapping.y = -int(mapping.x >> 31) | 0x80000000; // inverse function ((mapping.x >> 31) - 1) | 0x80000000;
		mapping.z = mapping.x ^ mapping.y;

		mapping.y = atomicMax(deltaQ, mapping.z);

		//printf("block[%5i]: delta=%e --\t%15u %15u -- old %15u\n", gidx, qdiff, mapping.x, mapping.z, mapping.y);
	}
	__syncthreads();
}

int cpu_Q_propagate(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	uint zero = 0;
	//cudaMemcpyToSymbol(deltaQmax, &zero, sizeof(uint), 0);

	// do the propagation
	gpu_Q_propagate<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Qn, cnv->d_P, cnv->d_PmQ, cnv->d_A0, cnv->d_deltaQmax);
	cudaError = cudaGetLastError();
	//if(cudaError != cudaSuccess) printf("Q error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();

	// copy back the maximum deltaQ in a voxel patch
	cudaError = cudaMemcpy(&cnv->deltaQmax, cnv->d_deltaQmax, sizeof(uint), cudaMemcpyDeviceToHost);
	//if(cudaError != cudaSuccess) printf("Q copy: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	// convert the deltaQmax into a flow that we can compare to a threshold
	uint tmpf = ((cnv->deltaQmax >> 31) - 1) | 0x80000000;
	tmpf = cnv->deltaQmax ^ tmpf;
	float dq = ((float*)(&tmpf))[0];
	//printf("max dq: (copied back %15u) %15.8e -- %15u\n", cnv->deltaQmax, dq, tmpf);

	// temporary -- reset the deltaQ
	cudaMemcpy(cnv->d_deltaQmax, &zero, sizeof(uint), cudaMemcpyHostToDevice);

	// switch the pointers
	number *tmp = cnv->d_Q;
	cnv->d_Q = cnv->d_Qn;
	cnv->d_Qn = tmp;

	//printf("Q propagated.\n");

	// returns 1 if Q is below tolerance (converged)
	return (dq < cnv->dqTolerance);
}


/// Calculates the total of the
// this kernel might be suboptimal since it does not run with blocksize = 1024, while it could!
__global__ void gpu_Q_sum(number *Q, number *partials, number qtot) {

	__shared__ number shQ[B_3];
	__shared__ bool isLastBlockDone;

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3 + sidx;
	
	shQ[sidx] = Q[gidx];
	__syncthreads();

	uint inext, vnext;
	number psum = shQ[sidx];

	// do a parallel reduction
	for(uint stride=1; stride < B_3; stride*=2) {

		inext = sidx + stride;
		vnext = inext * (inext < B_3);
		psum += shQ[vnext] * (inext < B_3);
		__syncthreads();

		//if(sidx==0) printf("psum0 %5i: %f %f\n", gidx, shQ[vnext], psum);

		shQ[sidx] = psum;
		__syncthreads();
	}
	// now the first thread has the total sum... maybe

	// compute the block index
	gidx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	uint gsize = gridDim.x * gridDim.y * gridDim.z;

	// write the partial sum there
	if(sidx == 0) {

		partials[gidx] = psum;
		__threadfence();
		unsigned int value = atomicInc(&blockCount, gsize);
		isLastBlockDone = (value == gsize-1);
		//printf("block %5i partial %f\n", gidx, psum);
	}
	__syncthreads();

	// do the final sum over partials
	if(isLastBlockDone) {

		uint nloops = (uint)ceilf((float)gsize/B_3); // hard coded for the 3d block size
		//if(sidx == 0) printf("GPU nloops %i\n", nloops);
		
		number totalsum = 0;
		psum = 0;
		for(uint i=0; i<nloops; i++) {

			inext = sidx + i*B_3;
			vnext = inext * (inext < gsize);
			shQ[sidx] = partials[vnext] * (inext < gsize);
			psum = shQ[sidx];
			__syncthreads();

			for(uint stride=1; stride<B_3; stride *=2) {

				inext = sidx + stride;
				vnext = inext * (inext < B_3);
				psum += shQ[vnext] * (inext < B_3);
				__syncthreads();

				shQ[sidx] = psum;
				__syncthreads();
			}
			totalsum += psum;
		} // end loop over batches

		if(sidx == 0) {
			//printf("GPU sum %f, expected %f \n", totalsum, qtot);
			partials[0] = qtot / totalsum;
			blockCount = 0;
		}
	}
}

__global__ void gpu_Q_normalise(number *Q, number *factor) {

	__shared__ number f;

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3 + sidx;

	if(sidx == 0) {
		f = factor[0];
	}
	__syncthreads();

	Q[gidx] = Q[gidx] * f;
}

void cpu_Q_sum(Convolver *cnv, Cube *cube) {

	number total;
	cudaError_t cudaError;

	// do the propagation
	gpu_Q_sum<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_partials, (number)cube->molecule.qtot);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();

	
	// copy back the total
	cudaError = cudaMemcpy(&total, cnv->d_partials, sizeof(number), cudaMemcpyDeviceToHost);
	assert(cudaError == cudaSuccess);
	//printf("total e: %f\n", total);
	
	assert(total == total);

	/*
	number *Q = (number*)malloc(sizeof(number) * cnv->maxpts);
	cudaError = cudaMemcpy(Q, cnv->d_Q, sizeof(number) * cnv->maxpts, cudaMemcpyDeviceToHost);
	assert(cudaError == cudaSuccess);

	total=0;
	for(int i=0;i<cnv->maxpts;i++){
		total += Q[i];
	}
	printf("total e CPU: %f\n", total);
	free(Q);
	*/
	
	// renormalize
	gpu_Q_normalise<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_partials);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
}


/* ************************************************************************** */
/* *** DISCREPANCY CALCULATORS *** ****************************************** */

/// Calculates the total of the
// this kernel might be suboptimal since it does not run with blocksize = 1024, while it could!
__global__ void gpu_Q_diff(number *Q, number *Qref, number *partials) {

	__shared__ number shQ[B_3];
	__shared__ bool isLastBlockDone;

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3 + sidx;
	
	shQ[sidx] = Q[gidx];
	shQ[sidx] -= Qref[gidx];
	shQ[sidx] = fabsf(shQ[sidx]);
	__syncthreads();

	uint inext, vnext;
	number psum = shQ[sidx];

	// do a parallel reduction
	for(uint stride=1; stride < B_3; stride*=2) {

		inext = sidx + stride;
		vnext = inext * (inext < B_3);
		psum += shQ[vnext] * (inext < B_3);
		__syncthreads();

		//if(sidx==0) printf("psum0 %5i: %f %f\n", gidx, shQ[vnext], psum);

		shQ[sidx] = psum;
		__syncthreads();
	}
	// now the first thread has the total sum... maybe

	// compute the block index
	gidx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	uint gsize = gridDim.x * gridDim.y * gridDim.z;

	// write the partial sum there
	if(sidx == 0) {

		partials[gidx] = psum;
		__threadfence();
		unsigned int value = atomicInc(&blockCount, gsize);
		isLastBlockDone = (value == gsize-1);
		//printf("block %5i partial %f\n", gidx, psum);
	}
	__syncthreads();

	// do the final sum over partials
	if(isLastBlockDone) {

		uint nloops = (uint)ceilf((float)gsize/B_3); // hard coded for the 3d block size
		//if(sidx == 0) printf("GPU nloops %i\n", nloops);
		
		number totalsum = 0;
		psum = 0;
		for(uint i=0; i<nloops; i++) {

			inext = sidx + i*B_3;
			vnext = inext * (inext < gsize);
			shQ[sidx] = partials[vnext] * (inext < gsize);
			psum = shQ[sidx];
			__syncthreads();

			for(uint stride=1; stride<B_3; stride *=2) {

				inext = sidx + stride;
				vnext = inext * (inext < B_3);
				psum += shQ[vnext] * (inext < B_3);
				__syncthreads();

				shQ[sidx] = psum;
				__syncthreads();
			}
			totalsum += psum;
		} // end loop over batches

		if(sidx == 0) {
			//printf("GPU sum %f\n", totalsum);
			partials[0] = 1.0f / totalsum;
			blockCount = 0;
		}
	}
}

number cpu_Q_diff(Convolver *cnv, Cube *cube) {

	number total;
	cudaError_t cudaError;

	// copy the ref Q to gpu
	cudaMemcpy(cnv->d_Qn, cube->Q, sizeof(number) * cube->npts, cudaMemcpyHostToDevice);

	// compute difference
	gpu_Q_diff<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Qn, cnv->d_partials);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
	
	// the final sum is in d_partials[0]
	cudaMemcpy(&total, cnv->d_partials, sizeof(number), cudaMemcpyDeviceToHost);
	return total;
}


/*
__global__ void gpu_A0_propagate(number *PmQ, number *A0, number *A0out) {

	__shared__ number buffer[Bp1_3];

	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * c_blocksPerSide + blockIdx.z * c_blocksPerSide * c_blocksPerSide) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// this is the direction 
	short dx, dy, dz, sidx;
	
	// data destination address in shared memory - for main block of data
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	

	// load the main block of data in shared mem
	buffer[sidx] = A0[gidx];
	__syncthreads();

	
	// compute directions for block search - block size x must be at least 6!!!
	dx = (threadIdx.x == 0) - (threadIdx.x == 1);
	dy = (threadIdx.x == 2) - (threadIdx.x == 3);
	dz = (threadIdx.x == 4) - (threadIdx.x == 5);

	// compute global data offset - BUG: missing global block offset
	gidx = 0;
	// first do the block offset
	sidx = blockIdx.x + dx;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx;

	sidx = blockIdx.y + dy;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx * c_blocksPerSide;

	sidx = blockIdx.z + dz;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx * c_blocksPerSide * c_blocksPerSide;
	gidx *= B_3;

	// there is an addressing bug somewhere!!!!!!

	// now do the thread offset
	gidx +=  (dx == 0) * threadIdx.y + (dx == -1) * (B-1);
	gidx += ((dy == 0) * ( (dx != 0)*threadIdx.y + (dz != 0)*threadIdx.z ) + (dy == -1) * (B-1)) * B;
	gidx += ((dz == 0) * threadIdx.z + (dz == -1) * (B-1)) * B_2;

	// compute destination in shared memory
	sidx = 0;
	sidx +=   (dx == 0) * (threadIdx.y+1) + (dx == 1) * (B+1);
	sidx += (((dy == 0) * ( (dx != 0)*(threadIdx.y+1) + (dz != 0)*(threadIdx.z+1) ) + (dy == 1) * (B+1))) * Bp1;
	sidx += (((dz == 0) * (threadIdx.z+1) + (dz == 1) * (B+1))) * Bp1_2;

	// load the periphery
	if(threadIdx.x < 6)
		buffer[sidx] = A0[gidx];
	__syncthreads();


	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	//printf("c_blocksPerSide %i \n",c_blocksPerSide);
	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * c_blocksPerSide + blockIdx.z * c_blocksPerSide * c_blocksPerSide) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;


	// propagate the field
	number f0 = buffer[sidx];
	number field = buffer[sidx];
	number acc = field * field;
	field -= f0 * (2 * c_A0_diff + c_A0_loss[0]);
	field -= acc * (2*(f0 >= 0) - 1) * c_A0_loss[1];
	field -= acc * f0 * c_A0_loss[2];

	// +/- x
	field += c_A0_diff * (buffer[sidx+1] + buffer[sidx-1]);
	field += c_A0_diff * (buffer[sidx+Bp1] + buffer[sidx-Bp1]);
	field += c_A0_diff * (buffer[sidx+Bp1_2] + buffer[sidx-Bp1_2]);
	
	// generators
	field += PmQ[gidx] * c_A0_gen;
	A0out[gidx] = field;
}
void cpu_A0_propagate(Cube *cube) {

	cudaError_t cudaError;

	gpu_A0_propagate<<<cube->gpu_grid, cube->gpu_block>>>(cube->d_PmQ, cube->d_A0, cube->d_A0new);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
}
void cpu_A0_propagate_all(Cube *cube) {

	int ns = cube->nCubes;
	int shmem = Bp1_3 * sizeof(number);

	cudaStream_t stream[ns];
	cudaError_t cudaError[ns];


	for (int i = 0; i < ns; i ++) {
		cudaError[i] = cudaStreamCreate(&stream[i]);
		assert(cudaError[i] == cudaSuccess);
	}


	for (int i = 0; i < ns; i ++) {
		int offset = i * cube->npts;
		gpu_A0_propagate<<<cube->gpu_grid, cube->gpu_block, shmem, stream[i]>>>(cube->d_PmQ+offset, cube->d_A0+offset, cube->d_A0new+offset);
		cudaError[i] = cudaGetLastError();
		cudaMemcpyAsync(cube->A0+offset, cube->d_A0new+offset, sizeof(number) * cube->npts, cudaMemcpyDeviceToHost, stream[i]);
		printf("stream %i done.\n", i);
	}

	cudaDeviceSynchronize();
	printf("device synched\n");

	for (int i = 0; i < ns; i ++) {
		cudaError[i] = cudaStreamDestroy(stream[i]);
		assert(cudaError[i] == cudaSuccess);
	}
}


// TODO!
__global__ void gpu_A0_propagateF(number *PmQ, number *A0, number *A0out) {

	__shared__ number buffer[Bp1_3];

	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * c_blocksPerSideF.x + blockIdx.z * c_blocksPerSideF.x * c_blocksPerSideF.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// this is the direction 
	short dx, dy, dz, sidx;
	
	// data destination address in shared memory - for main block of data
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	

	// load the main block of data in shared mem
	buffer[sidx] = A0[gidx];
	__syncthreads();

	
	// compute directions for block search - block size x must be at least 6!!!
	dx = (threadIdx.x == 0) - (threadIdx.x == 1);
	dy = (threadIdx.x == 2) - (threadIdx.x == 3);
	dz = (threadIdx.x == 4) - (threadIdx.x == 5);

	// compute global data offset
	gidx = 0;
	// first do the block offset
	sidx = blockIdx.x + dx;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx;

	sidx = blockIdx.y + dy;
	sidx += ((sidx < 0) - (sidx == gridDim.y))*gridDim.y;
	gidx += sidx * c_blocksPerSideF.x;

	sidx = blockIdx.z + dz;
	sidx += ((sidx < 0) - (sidx == gridDim.z))*gridDim.z;
	gidx += sidx * c_blocksPerSideF.x * c_blocksPerSideF.y;
	gidx *= B_3;


	// now do the thread offset
	gidx +=  (dx == 0) * threadIdx.y + (dx == -1) * (B-1);
	gidx += ((dy == 0) * ( (dx != 0)*threadIdx.y + (dz != 0)*threadIdx.z ) + (dy == -1) * (B-1)) * B;
	gidx += ((dz == 0) * threadIdx.z + (dz == -1) * (B-1)) * B_2;

	// compute destination in shared memory
	sidx = 0;
	sidx +=   (dx == 0) * (threadIdx.y+1) + (dx == 1) * (B+1);
	sidx += (((dy == 0) * ( (dx != 0)*(threadIdx.y+1) + (dz != 0)*(threadIdx.z+1) ) + (dy == 1) * (B+1))) * Bp1;
	sidx += (((dz == 0) * (threadIdx.z+1) + (dz == 1) * (B+1))) * Bp1_2;

	// load the periphery
	if(threadIdx.x < 6)
		buffer[sidx] = A0[gidx];
	__syncthreads();


	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	//printf("c_blocksPerSide %i \n",c_blocksPerSide);
	
	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * c_blocksPerSideF.x + blockIdx.z * c_blocksPerSideF.x * c_blocksPerSideF.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;



	// propagate the field
	number f0 = buffer[sidx];
	number field = buffer[sidx];
	number acc = field * field;
	field -= f0 * (2 * c_A0_diff + c_A0_loss[0]);
	field -= acc * (2*(f0 >= 0) - 1) * c_A0_loss[1];
	field -= acc * f0 * c_A0_loss[2];

	// +/- x/y/z
	field += c_A0_diff * (buffer[sidx+1] + buffer[sidx-1]);
	field += c_A0_diff * (buffer[sidx+Bp1] + buffer[sidx-Bp1]);
	field += c_A0_diff * (buffer[sidx+Bp1_2] + buffer[sidx-Bp1_2]);
	
	// TODO: propagate also in diagonal directions?


	// generators
	field += PmQ[gidx] * c_A0_gen;
	A0out[gidx] = field;
}
void cpu_A0_propagateF(Cube *cube) {

	cudaError_t cudaError;

	gpu_A0_propagateF<<<cube->gpu_grid, cube->gpu_block>>>(cube->d_PmQ, cube->d_A0, cube->d_A0new);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
}
*/


/*
__global__ void gpu_Q_propagate(number *Q, number *Qout, number *P, number *PmQ, number *A0) {

	__shared__ number shQ[Bp1_3];
	__shared__ number shA[Bp1_3];

	// global memory address
	uint gidx = (blockIdx.x + blockIdx.y * c_blocksPerSide + blockIdx.z * c_blocksPerSide * c_blocksPerSide) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// this is the direction 
	short dx, dy, dz, sidx;
	
	// data destination address in shared memory - for main block of data
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	

	// load the main block of data in shared mem
	shQ[sidx] = Q[gidx];
	shA[sidx] = A0[gidx];
	__syncthreads();

	
	// compute directions for block search - block size x must be at least 6!!!
	dx = (threadIdx.x == 0) - (threadIdx.x == 1);
	dy = (threadIdx.x == 2) - (threadIdx.x == 3);
	dz = (threadIdx.x == 4) - (threadIdx.x == 5);

	// compute global data offset
	gidx = 0;
	// first do the block offset
	sidx = blockIdx.x + dx;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx;

	sidx = blockIdx.y + dy;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx * c_blocksPerSide;

	sidx = blockIdx.z + dz;
	sidx += ((sidx < 0) - (sidx == gridDim.x))*gridDim.x;
	gidx += sidx * c_blocksPerSide * c_blocksPerSide;
	gidx *= B_3;

	// there is an addressing bug somewhere!!!!!!

	// now do the thread offset
	gidx +=  (dx == 0) * threadIdx.y + (dx == -1) * (B-1);
	gidx += ((dy == 0) * ( (dx != 0)*threadIdx.y + (dz != 0)*threadIdx.z ) + (dy == -1) * (B-1)) * B;
	gidx += ((dz == 0) * threadIdx.z + (dz == -1) * (B-1)) * B_2;

	// compute destination in shared memory
	sidx = 0;
	sidx +=   (dx == 0) * (threadIdx.y+1) + (dx == 1) * (B+1);
	sidx += (((dy == 0) * ( (dx != 0)*(threadIdx.y+1) + (dz != 0)*(threadIdx.z+1) ) + (dy == 1) * (B+1))) * Bp1;
	sidx += (((dz == 0) * (threadIdx.z+1) + (dz == 1) * (B+1))) * Bp1_2;

	// load the periphery
	if(threadIdx.x < 6) {
		shQ[sidx] = Q[gidx];
		shA[sidx] = A0[gidx];
	}

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;
	//printf("c_blocksPerSide %i \n",c_blocksPerSide);
	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * c_blocksPerSide + blockIdx.z * c_blocksPerSide * c_blocksPerSide) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	__syncthreads();

	// trasfer charges based on the field
	number t, qfinal = 0;
	number q1 = shQ[sidx];
	
	
	// should be fixed now
	// transfer with x+1	
	t = tanhf(q1 * shA[sidx+1] - shQ[sidx+1] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx+1];
	qfinal += t;

	// transfer with x-1	
	t = tanhf(q1 * shA[sidx-1] - shQ[sidx-1] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx-1];
	qfinal += t;

	// transfer with y+1	
	t = tanhf(q1 * shA[sidx+Bp1] - shQ[sidx+Bp1] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx+Bp1];
	qfinal += t;

	// transfer with y-1	
	t = tanhf(q1 * shA[sidx-Bp1] - shQ[sidx-Bp1] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx-Bp1];
	qfinal += t;

	// transfer with z+1	
	t = tanhf(q1 * shA[sidx+Bp1_2] - shQ[sidx+Bp1_2] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx+Bp1_2];
	qfinal += t;

	// transfer with z-1	
	t = tanhf(q1 * shA[sidx-Bp1_2] - shQ[sidx-Bp1_2] * shA[sidx]);
	t *= (t > 0) * q1 + (t <= 0) * shQ[sidx-Bp1_2];
	qfinal += t;

	qfinal *= transferRate * 0.16f; // TODO: get the right transfer rate from parameters
	
	// compute the electronic diffusion
	t = -6*q1;
	t += (shQ[sidx+1] + shQ[sidx-1]);
	t += (shQ[sidx+Bp1] + shQ[sidx-Bp1]);
	t += (shQ[sidx+Bp1_2] + shQ[sidx-Bp1_2]);
	qfinal += t * diffusionRate * 0.16f;
	
	qfinal = q1 + qfinal;
	Qout[gidx] = qfinal;
	PmQ[gidx] = P[gidx] - qfinal;
}
void cpu_Q_propagate(Cube *cube) {

	cudaError_t cudaError;

	gpu_Q_propagate<<<cube->gpu_grid, cube->gpu_block>>>(cube->d_Q, cube->d_Qnew, cube->d_P, cube->d_PmQ, cube->d_A0);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();

	/* // switch the pointers
	number *tmp = cube->d_Q;
	cube->d_Q = cube->d_Qnew;
	cube->d_Qnew = tmp; * /
}



__device__ unsigned int blockCount = 0;
// this can be a linear block, covering all the Q and R?
__global__ void gpu_Q_compare(number *Q, number *R, number *output) {

	__shared__ number buffer[LB];
	__shared__ bool isLastBlockDone;

	// global memory address
	uint idx = threadIdx.x + blockIdx.x * LB;
	uint idxNext;

	// compute the difference result-reference
	buffer[threadIdx.x] = fabsf(Q[idx] - R[idx]);

	// do a parallel reduction
	number blocksum = buffer[threadIdx.x];
	__syncthreads();

	for(ushort stride=1; stride < LB; stride*=2) {

		idx = threadIdx.x + stride;
		idxNext = idx * (idx < LB);
		blocksum += buffer[idxNext] * (idx < LB);
		__syncthreads();

		buffer[threadIdx.x] = blocksum;
		__syncthreads();
	}

	if(threadIdx.x == 0) {

		output[blockIdx.x] = blocksum;
		__threadfence();

		uint value = atomicInc(&blockCount, gridDim.x);
		isLastBlockDone = (value == gridDim.x-1);
	}
	__syncthreads();


	if(isLastBlockDone) {

		uint nloops = (uint)ceilf((number)gridDim.x/LB);
		number totalsum = 0;
		for(uint i=0; i<nloops; i++) {

			idx = threadIdx.x + i*LB;
			idxNext = idx * (idx < gridDim.x);
			buffer[threadIdx.x] = output[idxNext] * (idx < gridDim.x);
			blocksum = buffer[threadIdx.x];

			for(ushort stride=1; stride<LB; stride *=2) {

				idx = threadIdx.x + stride;
				idxNext = idx*(idx < LB);
				blocksum += buffer[idxNext]*(idx < LB);
				__syncthreads();

				buffer[threadIdx.x] = blocksum;
				__syncthreads();
			}
			totalsum += blocksum;
		} // end loop over batches

		if(threadIdx.x == 0) {
			LB[0] = totalsum / (LB * gridDim.x);
			blockCount = 0;
		}
	}
}

void cpu_Q_compare(Cube *cube) {

	cudaError_t cudaError;

	gpu_Q_compare<<<cube->gpu_grid, cube->gpu_block>>>(cube->d_Q, ...);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
}

*/




/* *****************************************************************************
	
	Evaluation of the fitness of one element in the population:

	LOOP over reference molecules:
		

*/

void cpu_evaluate_element(Convolver *convo, int eID) {

	cudaError_t cudaError;


	// copy the parameters to constant memory **********************************

	cudaError = cudaMemcpyToSymbol(c_parameters, &convo->population[eID * DNASIZE], sizeof(number), 0, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	// *************************************************************************



	// loop over reference molecules
	for(int ri=0; ri<NREFS; ri++) {


		// reset the GPU memory

		// generate P on the GPU
		// one block of 8 threads for each atom in the molecule

		// copy P/Q


		// convo loop
		int converged = 0;
		while(converged == 0) {

			// propagate fields

			
			// propagate Q

			
			// check convergence


			converged = 1;
		}



	}



}










