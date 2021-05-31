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
#include "kernel-qube.h"

// TODO: these kernels are a bit slow


__device__ float gpu_SolidHarmonicR(short L, short m, float3 r) {

	float result = 0;
	result += (L == 0) * 0.28209479177387814f;

	result += (m == -1)*(L == 1) * 0.4886025119029199f  * r.y;
	result += (m ==  0)*(L == 1) * 0.4886025119029199f  * r.z;
	result += (m ==  1)*(L == 1) * 0.4886025119029199f  * r.x;
	
	result += (m == -2)*(L == 2) * 1.0925484305920792f  * r.x * r.y;
	result += (m == -1)*(L == 2) * 1.0925484305920792f  * r.z * r.y;
	result += (m ==  0)*(L == 2) * 0.31539156525252005f * (2*r.z*r.z - r.x*r.x - r.y*r.y);
	result += (m ==  1)*(L == 2) * 1.0925484305920792f  * r.x * r.z;
	result += (m ==  2)*(L == 2) * 0.5462742152960396f  * (r.x*r.x - r.y*r.y);

	//if(L > 0 && m < 0)
		//printf("Ylm %i %i %f %f %f = %f\n",L,m,r.x,r.y,r.z,result);

	return result;
}


__global__ void 
__launch_bounds__(512, 2)
gpu_densityqube_nosh(
	float 		dx,		 	// grid step size - in BOHR
	float3 		grid0, 		// grid origin in BOHR
	int 		natoms,		// number of atoms in the molecule
	float3*		coords, 	// atom coordinates in BOHR
	float*		alphas,
	float*		coeffs,
	int			norbs,		// number of orbitals
	short4*		almos,		// indexes of oorbital properties
	float*		dm,			// density matrix
	float*		qube 		// output density cube
	){

	__shared__ float3 scoords[100];

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < natoms) {
		scoords[sidx] = coords[sidx];
		//if(blockIdx.x == 0 && blockIdx.y == 0 & blockIdx.z == 0)
			//printf("atom %i = %f %f %f\n", sidx, scoords[sidx].x, scoords[sidx].y, scoords[sidx].z);
	}
	__syncthreads();

	//printf("%i\n", gidx);
	float charge = 0;

	// compute voxel position
	float3 voxpos;
	voxpos.x = grid0.x + (blockIdx.x * B + threadIdx.x) * dx + 0.5f*dx;
	voxpos.y = grid0.y + (blockIdx.y * B + threadIdx.y) * dx + 0.5f*dx;
	voxpos.z = grid0.z + (blockIdx.z * B + threadIdx.z) * dx + 0.5f*dx;


	// now we got the patch loaded
	for(ushort p=0; p<norbs; p++) {
		for(ushort q=0; q<=p; q++) {

			float partial = dm[p*norbs+q];
			
			float3 r = scoords[almos[p].x];
			r.x = voxpos.x - r.x;
			r.y = voxpos.y - r.y;
			r.z = voxpos.z - r.z;
			partial *= gpu_SolidHarmonicR(almos[p].y, almos[p].z, r);

			// multiply by the contracted gaussians
			r.x = r.x*r.x + r.y*r.y + r.z*r.z;
			r.y = 0;
			for(ushort ai=0; ai<MAXAOC; ai++) {
				//r.z = alphas[almos[p].w+ai];
				r.y += coeffs[almos[p].w+ai] * exp(-alphas[almos[p].w+ai] * r.x);
			}
			partial *= r.y;

			r = scoords[almos[q].x];
			r.x = voxpos.x - r.x;
			r.y = voxpos.y - r.y;
			r.z = voxpos.z - r.z;
			partial *= gpu_SolidHarmonicR(almos[q].y, almos[q].z, r);

			r.x = r.x*r.x + r.y*r.y + r.z*r.z;
			r.y = 0;
			for(ushort ai=0; ai<MAXAOC; ai++) {
				//r.z = alphas[almos[q].w+ai];
				r.y += coeffs[almos[q].w+ai] * exp(-alphas[almos[q].w+ai] * r.x);
			}
			partial *= r.y;

			if(p!=q) partial*=2;

			charge += partial;
		}
	}

	// this accounts for closed shell (2 electrons per orbital)
	// and multiplied by the voxel volume (integral of the wfn)
	charge = 2*charge*dx*dx*dx;

	// compute the write index

	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	qube[sidx] = charge;
}

__global__ void 
__launch_bounds__(512, 2)
gpu_densityqube_nosh_subgrid(
	float 		dx,		 	// grid step size - in BOHR
	float3 		grid0, 		// grid origin in BOHR
	int 		natoms,		// number of atoms in the molecule
	float3*		coords, 	// atom coordinates in BOHR
	float*		alphas,
	float*		coeffs,
	int			norbs,		// number of orbitals
	short4*		almos,		// indexes of oorbital properties
	float*		dm,			// density matrix
	float*		qube 		// output density cube
	){

	__shared__ float3 scoords[100];

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < natoms) {
		scoords[sidx] = coords[sidx];
		//if(blockIdx.x == 0 && blockIdx.y == 0 & blockIdx.z == 0)
			//printf("atom %i = %f %f %f\n", sidx, scoords[sidx].x, scoords[sidx].y, scoords[sidx].z);
	}
	__syncthreads();

	//printf("%i\n", gidx);
	float charge = 0;

	// compute voxel position
	float3 voxpos;
	//voxpos.x = grid0.x + (blockIdx.x * B + threadIdx.x) * dx + 0.5f*dx;
	//voxpos.y = grid0.y + (blockIdx.y * B + threadIdx.y) * dx + 0.5f*dx;
	//voxpos.z = grid0.z + (blockIdx.z * B + threadIdx.z) * dx + 0.5f*dx;


	// now we got the patch loaded
	for(ushort p=0; p<norbs; p++) {
		for(ushort q=0; q<=p; q++) {

			float dmelem = dm[p*norbs+q];

			// subgrid resolution
			for(ushort ix=0; ix<SUBGRID; ix++) {
				voxpos.x = grid0.x + (blockIdx.x * B + threadIdx.x) * dx +ix*dx*SUBGRIDDX + SUBGRIDDX2;
				for(ushort iy=0; iy<SUBGRID; iy++) {
					voxpos.y = grid0.y + (blockIdx.y * B + threadIdx.y) * dx + iy*dx*SUBGRIDDX + SUBGRIDDX2;
					for(ushort iz=0; iz<SUBGRID; iz++) {
						voxpos.z = grid0.z + (blockIdx.z * B + threadIdx.z) * dx + iz*dx*SUBGRIDDX + SUBGRIDDX2;


						float partial = dmelem;
						
						float3 r = scoords[almos[p].x];
						r.x = voxpos.x - r.x;
						r.y = voxpos.y - r.y;
						r.z = voxpos.z - r.z;
						partial *= gpu_SolidHarmonicR(almos[p].y, almos[p].z, r);

						// multiply by the contracted gaussians
						r.x = r.x*r.x + r.y*r.y + r.z*r.z;
						r.y = 0;
						for(ushort ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[almos[p].w+ai];
							r.y += coeffs[almos[p].w+ai] * exp(-alphas[almos[p].w+ai] * r.x);
						}
						partial *= r.y;

						r = scoords[almos[q].x];
						r.x = voxpos.x - r.x;
						r.y = voxpos.y - r.y;
						r.z = voxpos.z - r.z;
						partial *= gpu_SolidHarmonicR(almos[q].y, almos[q].z, r);

						r.x = r.x*r.x + r.y*r.y + r.z*r.z;
						r.y = 0;
						for(ushort ai=0; ai<MAXAOC; ai++) {
							//r.z = alphas[almos[q].w+ai];
							r.y += coeffs[almos[q].w+ai] * exp(-alphas[almos[q].w+ai] * r.x);
						}
						partial *= r.y;

						if(p!=q) partial*=2;

						charge += partial * SUBGRIDiV;
					}
				}
			}// end of subgrid loop
		}
	}

	// this accounts for closed shell (2 electrons per orbital)
	// and multiplied by the voxel volume (integral of the wfn)
	charge = 2*charge*dx*dx*dx;

	// compute the write index

	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	qube[sidx] = charge;
}


__global__ void 
__launch_bounds__(512, 4)
gpu_densityqube_shmem(
	float 		dx,		 	// grid step size - in BOHR
	float3 		grid0, 		// grid origin in BOHR
	int 		natoms,		// number of atoms in the molecule
	float3*		coords, 	// atom coordinates in BOHR
	float*		alphas,
	float*		coeffs,
	int			norbs,		// number of orbitals
	short4*		almos,		// indexes of oorbital properties
	float*		dm,			// density matrix
	float*		qube 		// output density cube
	){

	__shared__ float3 scoords[100];
	__shared__ float sDM[4096]; // B_3 * 8

	uint sidx = threadIdx.x + threadIdx.y * B + threadIdx.z * B_2;

	if(sidx < natoms) {
		scoords[sidx] = coords[sidx];
	}
	__syncthreads();

	//printf("%i\n", gidx);
	float charge = 0;

	// compute voxel position
	float3 voxpos;
	voxpos.x = grid0.x + (blockIdx.x * B + threadIdx.x) * dx + 0.5f*dx;
	voxpos.y = grid0.y + (blockIdx.y * B + threadIdx.y) * dx + 0.5f*dx;
	voxpos.z = grid0.z + (blockIdx.z * B + threadIdx.z) * dx + 0.5f*dx;

	ushort nDMbatch = (ushort)ceil((float)(norbs*norbs)/((float)B_3*8));
	ushort p = 0, q = 0;

	for(ushort batch=0; batch<nDMbatch; ++batch) {

		uint ridx = sidx + B_3*batch;
		#pragma unroll
		for(ushort k=0; k<8; k++) {
			if(ridx + k*B_3 < norbs*norbs) sDM[sidx+k*B_3] = dm[ridx+k*B_3];
			else sDM[sidx] = 0;
		}
		__syncthreads();

		for(ushort ib=0; ib<B_3*8; ++ib) {

			// compute dm for p/q element
			float partial = sDM[ib]; //[p*norbs+q];
			
			float3 r = scoords[almos[p].x];
			r.x = voxpos.x - r.x;
			r.y = voxpos.y - r.y;
			r.z = voxpos.z - r.z;
			partial *= gpu_SolidHarmonicR(almos[p].y, almos[p].z, r);

			// multiply by the contracted gaussians
			r.x = r.x*r.x + r.y*r.y + r.z*r.z;
			r.y = 0;
			for(ushort ai=0; ai<MAXAOC; ai++) {
				//r.z = alphas[almos[p].w+ai];
				r.y += coeffs[almos[p].w+ai] * exp(-alphas[almos[p].w+ai] * r.x);
			}
			partial *= r.y;

			r = scoords[almos[q].x];
			r.x = voxpos.x - r.x;
			r.y = voxpos.y - r.y;
			r.z = voxpos.z - r.z;
			partial *= gpu_SolidHarmonicR(almos[q].y, almos[q].z, r);

			r.x = r.x*r.x + r.y*r.y + r.z*r.z;
			r.y = 0;
			for(ushort ai=0; ai<MAXAOC; ai++) {
				//r.z = alphas[almos[q].w+ai];
				r.y += coeffs[almos[q].w+ai] * exp(-alphas[almos[q].w+ai] * r.x);
			}
			partial *= r.y;

			charge += partial;

			q++;
			if(q == norbs) {
				q = 0;
				p++;
			}
			if(p == norbs) break;
		}
		__syncthreads();
	}


	// this accounts for closed shell (2 electrons per orbital)
	// and multiplied by the voxel volume (integral of the wfn)
	charge = 2*charge*dx*dx*dx;

	// compute the write index

	sidx = (threadIdx.x + blockIdx.x*B);
	sidx+= (threadIdx.y + blockIdx.y*B) * gridDim.x * B;
	sidx+= (threadIdx.z + blockIdx.z*B) * gridDim.x * gridDim.y * B_2;
	qube[sidx] = charge;
}


void qm_densityqube(Molecule *m, Grid *g) {

	cudaError_t cudaError;
	printf("computing density qube (no shmem)...\n");

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	dim3 block(B,B,B);
	gpu_densityqube_nosh<<<g->GPUblocks, block>>>(
		g->step,
		g->origin,
		m->natoms,
		m->d_coords,
		m->basisset->d_alphas,
		m->basisset->d_coeffs,
		m->norbs,
		m->d_ALMOs,
		m->d_dm,
		g->d_qube
	);

	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_qube error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("kernel time: %lf \n", cpu_time_used);

	cudaError = cudaMemcpy(g->qube, g->d_qube, sizeof(float)*g->npts, cudaMemcpyDeviceToHost);assert(cudaError == cudaSuccess);
}
void qm_densityqube_subgrid(Molecule *m, Grid *g) {

	cudaError_t cudaError;
	printf("computing density qube (no shmem, subgrid 4)...\n");

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	dim3 block(B,B,B);
	gpu_densityqube_nosh_subgrid<<<g->GPUblocks, block>>>(
		g->step,
		g->origin,
		m->natoms,
		m->d_coords,
		m->basisset->d_alphas,
		m->basisset->d_coeffs,
		m->norbs,
		m->d_ALMOs,
		m->d_dm,
		g->d_qube
	);

	cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_qube error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("kernel time: %lf \n", cpu_time_used);

	cudaError = cudaMemcpy(g->qube, g->d_qube, sizeof(float)*g->npts, cudaMemcpyDeviceToHost);assert(cudaError == cudaSuccess);
}

void qm_densityqube_shmem(Molecule *m, Grid *g) {

	cudaError_t cudaError;
	printf("computing density qube...\n");


	clock_t start, end;
	double cpu_time_used;

	start = clock();


	dim3 block(B,B,B);
	gpu_densityqube_shmem<<<g->GPUblocks, block>>>(
		g->step,
		g->origin,
		m->natoms,
		m->d_coords,
		m->basisset->d_alphas,
		m->basisset->d_coeffs,
		m->norbs,
		m->d_ALMOs,
		m->d_dm,
		g->d_qube
	); cudaDeviceSynchronize();
	cudaError = cudaGetLastError();
	if(cudaError != cudaSuccess)
		printf("gpu_qube error: %s\n", cudaGetErrorString(cudaError));
	assert(cudaError == cudaSuccess);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("kernel time: %lf \n", cpu_time_used);

	cudaError = cudaMemcpy(g->qube, g->d_qube, sizeof(float)*g->npts, cudaMemcpyDeviceToHost); assert(cudaError == cudaSuccess);
}



void qm_grid_toGPU(Grid *g) {

	cudaError_t cudaError;
	cudaError = cudaMemcpy(g->d_qube, g->qube, sizeof(float)*g->npts*g->nfields, cudaMemcpyHostToDevice);
	assert(cudaError == cudaSuccess);
}
