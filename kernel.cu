
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


__global__ void gpu_cube_unwrap(number *Q, number *Qout) {

	uint ridx, widx;

	widx = threadIdx.x + blockIdx.x*B;
	widx+= (threadIdx.y + blockIdx.y*B)*gridDim.x*B;
	widx+= (threadIdx.z + blockIdx.z*B)*gridDim.x*gridDim.y*B_2;

	ridx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	ridx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[widx] = Q[ridx];
}

void cpu_cube_unwrap(Convolver *cnv, Cube *cube, number *gpusrc, number *gpudst) {

	cudaError_t cudaError;
	
	// wrap it and store in Q tensor
	gpu_cube_unwrap<<<cube->gpu_grid, cnv->gpu_block>>>(gpusrc, gpudst);
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_cube_unwrap error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
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
		cnv->d_Ve,
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




/* *** VNe CALCULATOR *** *************************************************** */

// one thread for each voxel
__global__ void gpu_VNN_make(
	float gridStep, 	// grid step size
	number *VNN, 
	int natoms,
	float3 *coords, 
	int *Zs) {

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3 + sidx;
	
	float vnn = 0;

	// compute voxel position
	
	__shared__ float3 atom;
	__shared__ int Z;

	float3 voxpos;
	voxpos.x = (blockIdx.x * B + threadIdx.x + 0.5f) * gridStep;
	voxpos.y = (blockIdx.y * B + threadIdx.y + 0.5f) * gridStep;
	voxpos.z = (blockIdx.z * B + threadIdx.z + 0.5f) * gridStep;
	float r;

	// for each atom
	for(int i=0; i<natoms; i++) {

		if(sidx == 0) {
			atom = coords[i];
			Z = Zs[i];
		}
		__syncthreads();

		r = (voxpos.x - atom.x) * (voxpos.x - atom.x);
		r+= (voxpos.y - atom.y) * (voxpos.y - atom.y);
		r+= (voxpos.z - atom.z) * (voxpos.z - atom.z);

		//r = Z * exp(-r);
		r = Z * rsqrt(r);
		vnn += r;
		__syncthreads();
	}


	VNN[gidx] = vnn;
}

void convolver_makeVNN(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	// the kernel needs to know the position of the atoms
	// relative to the origin of their grid

	cudaError = cudaMemcpy(cnv->d_coords, cube->molecule.coords, sizeof(float3)*cube->molecule.natoms, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpy(cnv->d_Zs, cube->molecule.Zs, sizeof(int)*cube->molecule.natoms, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);


	gpu_VNN_make<<<cube->gpu_grid, cnv->gpu_block>>>(
		DS,
		cnv->d_A0,
		cube->molecule.natoms,
		cnv->d_coords,
		cnv->d_Zs
	);
	cudaError = cudaGetLastError();
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



/* *** PmQ FIXER *** ******************************************************** */

__global__ void gpu_PmQ_make(number *Q, number *P, number *PmQ) {

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	uint gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3 + sidx;
	
	PmQ[gidx] = P[gidx] - Q[gidx];
}

void convolver_makePmQ(Convolver *cnv, Cube *cube) {

	number total;
	cudaError_t cudaError;

	// copy the ref Q to gpu
	cudaMemcpy(cnv->d_Q, cube->Q, sizeof(number) * cube->npts, cudaMemcpyHostToDevice);

	// compute difference
	gpu_PmQ_make<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_P, cnv->d_PmQ);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
}

/* ************************************************************************** */





/* *** FIELD PROPAGATOR *** ************************************************* */

__global__ void gpu_A0_propagate(number *Q, number *A0, number *A0out) {

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
	field += Q[gidx] * c_parameters[PARAM_A0_AGEN]; // add the field generation/adsorption from charges

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
}
__global__ void gpu_A0_propagate_tally(number *PmQ, number *A0, number *A0out, uint *delta) {

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
	sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += sidx;

	// compute the final value of the filed in the voxel
	number field = f0 + Adiff; // adds the diffusion
	// now we have the field that is left 


	field += PmQ[gidx] * c_parameters[PARAM_A0_AGEN]; // add the field generation/adsorption from charges

	// whatever is left can be lost to the bath
	f0 = fabsf(field);
	number loss;
	loss  = exp(-c_parameters[PARAM_A0_LOS1] * f0);
	loss += exp(-c_parameters[PARAM_A0_LOS2] * f0*f0);
	loss += exp(-c_parameters[PARAM_A0_LOS3] * f0*f0*f0);
	loss = 1.0f - loss / 3.0f; // divide by 3
	field -= loss*field;
	



	A0out[gidx] = field; // save the output

	// compute difference
	field = fabsf(field - buffer[sidx]);
	__syncthreads();

	buffer[sidx] = field;
	__syncthreads();

	
	// sum them all up
	for(unsigned short stride=1; stride < B_3; stride*=2) {

		dx = sidx + stride;
		dz = (dx < B_3);
		dy = dx * dz;
		field += buffer[dy] * dz;
		__syncthreads();

		buffer[sidx] = field;
		__syncthreads();
	}

	// now we have a partial sum - first thread writes the partial
	if(sidx == 0) {

		mapping.x = ((uint*)(&field))[0];
		mapping.y = -int(mapping.x >> 31) | 0x80000000; // inverse function ((mapping.x >> 31) - 1) | 0x80000000;
		mapping.z = mapping.x ^ mapping.y;

		mapping.y = atomicMax(delta, mapping.z);
	}
}

void cpu_A0_propagate(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	gpu_A0_propagate<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_A0, cnv->d_A0n);
	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	//printf("A error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	number *tmp = cnv->d_A0;
	cnv->d_A0 = cnv->d_A0n;
	cnv->d_A0n = tmp;

	
	//printf("A0 propagated.\n");
}
number cpu_A0_propagate_tally(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	gpu_A0_propagate_tally<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_PmQ, cnv->d_A0, cnv->d_A0n, cnv->d_deltaQmax);
	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);

	// copy back the maximum delta in a voxel patch
	cudaError = cudaMemcpy(&cnv->deltaQmax, cnv->d_deltaQmax, sizeof(uint), cudaMemcpyDeviceToHost);
	assert(cudaError == cudaSuccess);

	// convert the deltaQmax into a flow that we can compare to a threshold
	uint tmpf = ((cnv->deltaQmax >> 31) - 1) | 0x80000000;
	tmpf = cnv->deltaQmax ^ tmpf;
	float dq = ((float*)(&tmpf))[0] / B_3;

	#ifdef DEBUGPRINT
	printf("max deltaA0: (copied back %15u) %15.8e -- %15u\n", cnv->deltaQmax, dq, tmpf);
	#endif

	// temporary -- reset the deltaQ
	uint zero = 0;
	cudaMemcpy(cnv->d_deltaQmax, &zero, sizeof(uint), cudaMemcpyHostToDevice);


	number *tmp = cnv->d_A0;
	cnv->d_A0 = cnv->d_A0n;
	cnv->d_A0n = tmp;

	return dq;
}




__global__ void gpu_Vee_propagate(number *Q, number *A0, number *A0out) {

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

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;

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

	Adiff = Adiff * OneOverDIFFTOT - f0;
	Adiff *= c_parameters[PARAM_A0_DIFF];


	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	// compute the final value of the filed in the voxel
	number field = f0 + Adiff; // adds the diffusion
	field += Q[gidx] * c_parameters[PARAM_A0_AGEN]; // add the field generation/adsorption from charges

	// whatever is left can be lost to the bath
	f0 = fabsf(field);
	number loss;
	loss  = exp(-c_parameters[PARAM_A0_LOS1] * f0);
	loss += exp(-c_parameters[PARAM_A0_LOS2] * f0*f0);
	loss += exp(-c_parameters[PARAM_A0_LOS3] * f0*f0*f0);
	loss = 1.0f - loss/3.0f; // divide by 3
	field -= loss*field;
	
	// save the output
	A0out[gidx] = field;
}

void cpu_Vee_propagate(Convolver *cnv, Cube *cube) {

	cudaError_t cudaError;

	gpu_Vee_propagate<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Ve, cnv->d_A0n);
	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	//printf("A error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	number *tmp = cnv->d_Ve;
	cnv->d_Ve = cnv->d_A0n;
	cnv->d_A0n = tmp;

	
	//printf("A0 propagated.\n");
}

/* ************************************************************************** */





/* *** CHARGE PROPAGATOR *** ************************************************ */

__global__ void gpu_Q_propagate_OLD(number *Q, number *Qout, number *A0, number *deltaQ) {

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
	

	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[gidx] = qfinal;
	//PmQ[gidx] = P[gidx] - qfinal;


	// compute the change in charge inside this block
	//gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
	qdiff = fabsf(q1 - qfinal);
	sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2; // reindexing of shared mem
	shQ[sidx] = qdiff;
	__syncthreads();

	// sum them all up
	for(unsigned short stride=1; stride < B_3; stride*=2) {

		dx = sidx + stride;
		dz = (dx < B_3);
		dy = dx * dz;
		qdiff += shQ[dy] * dz;
		__syncthreads();

		shQ[sidx] = qdiff;
		__syncthreads();
	}


	// now we have a partial sum - first thread writes the partial
	// use atomics to accumulate

	if(sidx == 0) {

		atomicAdd(deltaQ, shQ[0]);

		/*
		mapping.x = ((uint*)(&qdiff))[0];
		mapping.y = -int(mapping.x >> 31) | 0x80000000; // inverse function ((mapping.x >> 31) - 1) | 0x80000000;
		mapping.z = mapping.x ^ mapping.y;

		mapping.y = atomicMax(deltaQ, mapping.z);
		*/
		//printf("block[%5i]: delta=%e --\t%15u %15u -- old %15u\n", gidx, qdiff, mapping.x, mapping.z, mapping.y);
	}
	//__syncthreads();
}


// only charge propagation
__global__ void gpu_Q_propagate(number *Q, number *Qout, number *A0, number *Ve) {

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
	shA[sidx] = A0[gidx] - Ve[gidx];
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
		shA[sidx] = A0[gidx] - Ve[gidx];
		shQ[sidx] = Q[gidx];
	}
	__syncthreads();

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;

	// trasfer charges based on the field
	number t, qfinal = 0;
	number q1 = shQ[sidx];
	number field1 = shA[sidx];
	number qdiff = 0, qtrans = 0;
	number factor, ksum = 0;


	#pragma unroll
	for(dx=-1; dx<=1; ++dx) {
		#pragma unroll
		for(dy=-1; dy<=1; ++dy) {
			#pragma unroll
			for(dz=-1; dz<=1; ++dz) {

				gidx = sidx + dx + dy*Bp1 + dz*Bp1_2;

				mapping.z = abs(dx) + abs(dy) + abs(dz);
				factor = (mapping.z == 1) + (mapping.z == 2) * OneOverSqrt2 + (mapping.z == 3) * OneOverSqrt3;

				// calculate how much is transferred from here to the neighbour
				t = tanhf((q1*shA[gidx] - shQ[gidx]*field1 + c_parameters[PARAM_QQ_TRNS] * q1)) * factor;
				//t = tanhf((shA[gidx] - field1)*c_parameters[PARAM_QQ_TRNS]) * factor;

				// positive => from here to neighbour => scale by my charge
				// negative => from neighbour to here => scale by its charge
				t *= (t > 0) * q1 + (t < 0) * shQ[gidx];

				
				ksum += t;
			}
		}
	}

	// this is the total chage that left the voxel
	ksum *= OneOverDIFFTOT * c_parameters[PARAM_QQ_DIFF];
	// and this is the final value of the charge
	qfinal = q1 - ksum;

	
	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[gidx] = qfinal;
}

// this also compute charge change and total charge
__global__ void gpu_Q_propagate_wstats(number *Q, number *Qout, number *A0, number *Ve, number *deltaQ) {

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
	shA[sidx] = A0[gidx] - Ve[gidx];
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
		shA[sidx] = A0[gidx] - Ve[gidx];
		shQ[sidx] = Q[gidx];
	}
	__syncthreads();

	// restore the shmem address of this thread
	sidx = (threadIdx.x+1) + (threadIdx.y+1) * Bp1 + (threadIdx.z+1) * Bp1_2;

	// trasfer charges based on the field
	number t, qfinal = 0;
	number q1 = shQ[sidx];
	number field1 = shA[sidx];
	number qdiff = 0, qtrans = 0;
	number factor, ksum = 0;


	#pragma unroll
	for(dx=-1; dx<=1; ++dx) {
		#pragma unroll
		for(dy=-1; dy<=1; ++dy) {
			#pragma unroll
			for(dz=-1; dz<=1; ++dz) {

				gidx = sidx + dx + dy*Bp1 + dz*Bp1_2;

				mapping.z = abs(dx) + abs(dy) + abs(dz);
				factor = (mapping.z == 1) + (mapping.z == 2) * OneOverSqrt2 + (mapping.z == 3) * OneOverSqrt3;

				// calculate how much is transferred from here to the neighbour
				t = tanhf((q1*shA[gidx] - shQ[gidx]*field1 + c_parameters[PARAM_QQ_TRNS] * q1)) * factor;
				//t = tanhf((shA[gidx] - field1)*c_parameters[PARAM_QQ_TRNS]) * factor;

				// positive => from here to neighbour => scale by my charge
				// negative => from neighbour to here => scale by its charge
				t *= (t > 0) * q1 + (t < 0) * shQ[gidx];

				
				ksum += t;
			}
		}
	}

	// this is the total chage that left the voxel
	ksum *= OneOverDIFFTOT * c_parameters[PARAM_QQ_DIFF];
	// and this is the final value of the charge
	qfinal = q1 - ksum;

	
	// global address for writing - recalculated from previous
	gidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) * B_3;
	gidx += threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	Qout[gidx] = qfinal;

	// compute the change in charge inside this block
	// also compute the total charge
	sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2; // reindexing of shared mem

	qdiff = fabsf(q1 - qfinal);
	shQ[sidx] = qfinal;
	shA[sidx] = qdiff;
	__syncthreads();

	// sum them all up
	for(unsigned short stride=1; stride < B_3; stride*=2) {

		dx = sidx + stride;
		dz = (dx < B_3);
		dy = dx * dz;
		qfinal += shQ[dy] * dz;
		qdiff  += shA[dy] * dz;
		__syncthreads();

		shQ[sidx] = qfinal;
		shA[sidx] = qdiff;
		__syncthreads();
	}

	// now we have a partial sum - first thread writes the partial
	// use atomics to accumulate

	if(sidx == 1) shQ[1] = shA[0];
	__syncthreads();

	if(sidx <= 1) {
		atomicAdd(&deltaQ[sidx], shQ[sidx]);
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


number cpu_Q_propagate(Convolver *cnv, Cube *cube, int wstats) {

	cudaError_t cudaError;
	number zeros[2] = {0,0};
	

	if(wstats == 1) {
		cudaMemcpy(cnv->d_partials, zeros, sizeof(number) * 2, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		gpu_Q_propagate_wstats<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Qn, cnv->d_A0, cnv->d_Ve, cnv->d_partials);

	} else {

		gpu_Q_propagate<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Qn, cnv->d_A0, cnv->d_Ve);

	}
	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess) printf("Q error [stats %i]: %s\n", wstats, cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);


	if(wstats == 1) {

		cudaMemcpy(zeros, cnv->d_partials, sizeof(number) * 2, cudaMemcpyDeviceToHost);
		zeros[1] /= cube->npts; // Q diff

		// use qtot to normalize
		zeros[0] = ((number)cube->molecule.qtot) / zeros[0];
		cudaMemcpy(cnv->d_partials, zeros, sizeof(number) * 1, cudaMemcpyDeviceToHost);
		
		gpu_Q_normalise<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Qn, cnv->d_partials);
		cudaDeviceSynchronize();
	} else {
		zeros[1] = 1;
	}

	//printf("Q propagate output: %15.8e -- %15.8e\n", dqtot[0], dqtot[1]);

	// switch the pointers
	number *tmp = cnv->d_Q;
	cnv->d_Q = cnv->d_Qn;
	cnv->d_Qn = tmp;

	//printf("Q propagated.\n");
	return zeros[1];
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
			partials[0] = totalsum;
			blockCount = 0;
		}
	}
}

__global__ void gpu_Q_diff_atomic(number *Q, number *Qref, number *partials) {

	__shared__ number shQ[B_3];

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

		shQ[sidx] = psum;
		__syncthreads();
	}
	// now the first thread has the total sum... maybe

	// compute the block index
	gidx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
	uint gsize = gridDim.x * gridDim.y * gridDim.z;

	// write the partial sum there
	if(sidx == 0) {
		atomicAdd(partials, psum);
	}
}



number cpu_Q_diff(Convolver *cnv, Cube *cube) {

	number total = 0;
	cudaError_t cudaError;

	// copy the ref Q to gpu
	cudaMemcpy(cnv->d_Qn, cube->Q, sizeof(number) * cube->npts, cudaMemcpyHostToDevice);
	cudaMemcpy(cnv->d_partials, &total, sizeof(number), cudaMemcpyHostToDevice);


	// compute difference
	gpu_Q_diff_atomic<<<cube->gpu_grid, cnv->gpu_block>>>(cnv->d_Q, cnv->d_Qn, cnv->d_partials);
	cudaError = cudaGetLastError();
	assert(cudaError == cudaSuccess);
	cudaDeviceSynchronize();
	
	// the final sum is in d_partials[0]
	cudaMemcpy(&total, cnv->d_partials, sizeof(number), cudaMemcpyDeviceToHost);
	return total / cube->npts;
}

/* ************************************************************************** */










