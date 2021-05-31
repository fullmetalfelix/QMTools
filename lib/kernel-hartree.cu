#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "qmtools.h"
#include "molecule.h"
#include "basisset.h"
#include "vectors.h"


// TODO: these kernels are a bit slow





__global__ void gpu_hartree(
	float 		q_dx,	 	// density grid parameters
	float3 		q_x0, 		// 
	dim3 		q_n, 		// 
	float* 		q,			// density grid

	float 		V_dx, 		// hartree grid parameters
	float3 		V_x0,
	float*		V, 			// output hartree qube
	
	int 		natoms,		// number of atoms in the molecule
	int* 		types,
	float3*		coords 		// atom coordinates in BOHR
){

	__shared__ int styp[100];
	__shared__ float3 scoords[100];
	__shared__ float sQ[B_3];


	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < natoms) {
		styp[sidx] = types[sidx];
		scoords[sidx] = coords[sidx];
	}
	__syncthreads();

	float hartree = 0;

	// compute voxel position in the output V grid
	float3 V_pos;
	V_pos.x = V_x0.x + (blockIdx.x * B + threadIdx.x) * V_dx + 0.5f*V_dx;
	V_pos.y = V_x0.y + (blockIdx.y * B + threadIdx.y) * V_dx + 0.5f*V_dx;
	V_pos.z = V_x0.z + (blockIdx.z * B + threadIdx.z) * V_dx + 0.5f*V_dx;


	// loop over the blocks of density grid
	for(ushort x=0; x<q_n.x; ++x) {
		for(ushort y=0; y<q_n.y; ++y) {
			for(ushort z=0; z<q_n.z; ++z) {

				// load a patch of Q grid
				uint ridx;
				ridx = x*B + threadIdx.x;
				ridx+=(y*B + threadIdx.y) * q_n.x * B;
				ridx+=(z*B + threadIdx.z) * q_n.x*q_n.y*B_2;
				sQ[sidx] = q[ridx];
				__syncthreads();
				// now we have the patch... loop!


				for(ushort sx=0; sx<B; ++sx) {
					for(ushort sy=0; sy<B; ++sy) {
						for(ushort sz=0; sz<B; ++sz) {

							float c, r=0; // distance between the V evaluation point and the q voxel center

							c = V_pos.x - (q_x0.x + (x*B + sx) * q_dx + 0.5f*q_dx);
							r = c*c;
							c = V_pos.y - (q_x0.y + (y*B + sy) * q_dx + 0.5f*q_dx);
							r+= c*c;
							c = V_pos.z - (q_x0.z + (z*B + sz) * q_dx + 0.5f*q_dx);
							r+= c*c;
							r = sqrtf(r);

							/*r.x = V_pos.x - (q_x0.x + (x*B + sx) * q_dx + 0.5f*q_dx);
							r.y = V_pos.y - (q_x0.y + (y*B + sy) * q_dx + 0.5f*q_dx);
							r.z = V_pos.z - (q_x0.z + (z*B + sz) * q_dx + 0.5f*q_dx);
							r.w = r.x*r.x + r.y*r.y + r.z*r.z;
							r.w = sqrtf(r.w);*/

							if(r < 0.5f*q_dx) 
								hartree += (sQ[sx + sy*B + sz*B_2]/q_dx) * (3.0f - 4.0f*r*r/(q_dx*q_dx));
							else 
								hartree += sQ[sx + sy*B + sz*B_2] / r;
						}
					}
				}
				__syncthreads();
			}
		}
	}

	hartree = -hartree; // because electrons are negative!

	// add the nuclear potential
	for(ushort i=0; i<natoms; i++) {

		float c, r=0; // distance between the V evaluation point and the q voxel center

		c = V_pos.x - scoords[i].x; r = c*c;
		c = V_pos.y - scoords[i].y; r+= c*c;
		c = V_pos.z - scoords[i].z; r+= c*c;
		r = sqrtf(r);
		hartree += styp[i] / r;
	}

	// write results
	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	V[sidx] = hartree;
}



void qm_hartree(Molecule *m, Grid *q, Grid *v) {

	cudaError_t cudaError;
	printf("computing hartree qube...\n");

	dim3 block(B,B,B);
	gpu_hartree<<<v->GPUblocks, block>>>(
		q->step,
		q->origin,
		q->GPUblocks,
		q->d_qube,

		v->step,
		v->origin,
		v->d_qube,

		m->natoms,
		m->d_types,
		m->d_coords
	);

	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_v_qube error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);
	// TODO: THERE IS AN INVALID MEMORY ADDRESS IN THE KERNEL!?

	cudaError = cudaMemcpy(v->qube, v->d_qube, sizeof(float)*v->npts, cudaMemcpyDeviceToHost);assert(cudaError == cudaSuccess);
}

