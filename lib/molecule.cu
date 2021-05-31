#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "qmtools.h"
#include "basisset.h"
#include "molecule.h"
#include "vectors.h"



/// Computes the grid properties
// coordinates must already be in bohr at this point
// no allocation is done here
void molecule_init(Molecule *mol) {

	int natm = mol->natoms;
	int norbs = mol->norbs;

	cudaError_t cudaError;
	cudaError = cudaMalloc((void**)&mol->d_types, sizeof(int)*natm); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_coords, sizeof(float3)*natm); assert(cudaError == cudaSuccess);

	cudaError = cudaMemcpy(mol->d_types, mol->types, sizeof(int)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(mol->d_coords, mol->coords, sizeof(float3)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);


	cudaError = cudaMalloc((void**)&mol->d_dm, sizeof(float)*norbs*norbs); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(mol->d_dm, mol->dm, sizeof(float)*norbs*norbs, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	cudaError = cudaMalloc((void**)&mol->d_ALMOs, sizeof(short4)*norbs); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(mol->d_ALMOs, mol->ALMOs, sizeof(short4)*norbs, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);
}



void molecule_del(Molecule *m) {

	cudaError_t cudaError;
	cudaError = cudaFree(m->d_types); assert(cudaError == cudaSuccess);
	cudaError = cudaFree(m->d_coords); assert(cudaError == cudaSuccess);
	cudaError = cudaFree(m->d_dm); assert(cudaError == cudaSuccess);
	cudaError = cudaFree(m->d_ALMOs); assert(cudaError == cudaSuccess);
}


// NOT USED!
// this function has to create a Grid from scratch and return it
void molecule_densitygrid(Molecule *m, Grid *g, float step, float fat) {


	float3 crdmax = make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX);
	float3 crdmin = make_float3( FLT_MAX, FLT_MAX, FLT_MAX);

	for(int i=0; i<m->natoms; i++) {
		
		crdmax = float3_max(crdmax, m->coords[i]);
		crdmin = float3_min(crdmin, m->coords[i]);
	}

	crdmin = crdmin -fat * ANG2BOR;
	crdmax = crdmax +fat * ANG2BOR;
	
	float3 grd = crdmax-crdmin;
	g->origin = crdmin;

	//printf("grid real size: %f %f %f\n", grd.x, grd.y, grd.z);
	//printf("grid origin: %f %f %f\n", crdmin.x, crdmin.y, crdmin.z);

	grd = grd / step;
	grd = grd / 8.0f;
	grd = float3_ceiled(grd);
	grd = grd * 8;
	
	g->shape = dim3(grd.x, grd.y, grd.z);
	g->GPUblocks = dim3(g->shape.x / 8, g->shape.y / 8, g->shape.z / 8);
	g->npts = g->shape.x * g->shape.y * g->shape.z;


	//printf("grid shape: %i %i %i - total: %i\n", mol->grid.x,mol->grid.y,mol->grid.z,mol->npts);
	//printf("grid blocks: %i %i %i\n", mol->blocks.x,mol->blocks.y,mol->blocks.z);

}



/*

void molecule_gpu_init(Molecule *mol) {

	cudaError_t cudaError;
	int natm = mol->natoms;

	// atomic types
	int *d_Zs;
	cudaError = cudaMalloc((void**)&d_Zs, sizeof(int)*natm); assert(cudaError == cudaSuccess);
	cudaMemcpy(d_Zs, mol->types, sizeof(int)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);
	mol->d_Zs = d_Zs;

	// coordinates
	float3 *d_coords, *coords;
	coords = (float3*)malloc(sizeof(float3)*natm);
	memcpy(coords, mol->coords, sizeof(float3)*natm);
	for(int i=0; i<mol->natoms; i++) // move every atom so that grid0 is 0!
		coords[i] = coords[i] - mol->grid0;
	cudaError = cudaMalloc((void**)&d_coords, sizeof(float3)*natm); assert(cudaError == cudaSuccess);
	cudaMemcpy(d_coords, coords, sizeof(float3)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);
	memcpy(mol->coords, coords, sizeof(float3) * natm);
	mol->d_coords = d_coords;
	free(coords);

	cudaError = cudaMalloc((void**)&mol->d_refQ, sizeof(float)*mol->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_nnQ,  sizeof(float)*mol->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_VNe,  sizeof(float)*mol->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_VNe_az, sizeof(float)*mol->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_VNe_az_ay, sizeof(float)*mol->grid.x*mol->grid.y); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&mol->d_VNe_az_ay_ax, sizeof(float)*mol->grid.x); assert(cudaError == cudaSuccess);
}


void molecule_gpu_free(Molecule *mol) {
	cudaFree(mol->d_Zs);
	cudaFree(mol->d_coords);
	cudaFree(mol->d_refQ);
	cudaFree(mol->d_nnQ);
	cudaFree(mol->d_VNe);
	cudaFree(mol->d_VNe_az);
	cudaFree(mol->d_VNe_az_ay);
	cudaFree(mol->d_VNe_az_ay_ax);
}



void molecule_write_bin(Molecule *mol, FILE *fbin) {

	fwrite(&mol->natoms, sizeof(int), 1, fbin);
	for(int i=0; i<mol->natoms; i++) {
		fwrite(&mol->types[i], sizeof(int), 1, fbin);
		fwrite(&mol->coords[i], sizeof(float3), 1, fbin);
	}
}
*/