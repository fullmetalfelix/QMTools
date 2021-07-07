#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "qmtools.h"

#include "basisset.h"
#include "molecule.h"


/*#include "kernel-rpts.h"
#include "kernel-acsf.h"
#include "kernel-qube.h"*/


int nTypes = 10;
int types[] = {1,6,7,8,9,14,15,16,17,35};
__constant__ int c_types[NTYPES];


//Initialization of QMTools, copy types to device memory
void qm_ini(QMTools *obj) {

	printf("ntypes: %i\n", NTYPES);
	for(int i=0; i<NTYPES; i++) {
		printf("%i ", types[i]);
	}
	printf("\n");
	cudaMemcpyToSymbol(c_types, types, sizeof(float)*NTYPES);
}



void qm_del(QMTools *obj) {

	//scsf_curand_free(obj);
}


//Initialization of Grid variables, copy qube and variables npts and nfields to device memory

void qm_grid_ini(Grid *g) {

	cudaError_t cudaError;
	cudaError = cudaMalloc((void**)&g->d_qube, sizeof(float)*g->npts*g->nfields); assert(cudaError == cudaSuccess);
}

void qm_grid_del(Grid *obj) {

	cudaError_t cudaError;
	cudaError = cudaFree(obj->d_qube); assert(cudaError == cudaSuccess);
}


//Write Grid and Molecule variables to .bin file

void qm_gridmol_write(Grid *g, Molecule *m, const char* filename) {

	FILE *fbin = fopen(filename, "wb");

	fwrite(&m->natoms, sizeof(int), 1, fbin);
	fwrite(m->types, sizeof(int), m->natoms, fbin);
	fwrite(m->coords, sizeof(float3), m->natoms, fbin);

	fwrite(&g->origin, sizeof(float3), 1, fbin);
	fwrite(&g->shape, sizeof(dim3), 1, fbin);
	fwrite(&g->npts, sizeof(uint), 1, fbin);
	fwrite(&g->step, sizeof(float), 1, fbin);
	printf("npts: %i",g->npts);
	fwrite(g->qube, sizeof(float), g->npts, fbin);
	printf("oh yea\n");
	
	fclose(fbin);
}


//???

 SCSFGPU* scsf_gpu_allocate(QMTools *obj, Molecule *m) {

 	SCSFGPU *g = (SCSFGPU*)malloc(sizeof(SCSFGPU));
 	/*
	cudaError_t cudaError;
	int natm = m->natoms;
	int norb = m->norbs;
	int nrpts = obj->nrpts;
	int nacsf = obj->nacsf;

	// constants to constant memory
	int ns[2] = {natm, 0};
	cudaMemcpyToSymbol(c_esph_ns, ns, sizeof(int)*2);


	// atomic types
	cudaError = cudaMalloc((void**)&g->Zs, sizeof(int)*natm); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(g->Zs, m->types, sizeof(int)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	// coordinates
	cudaError = cudaMalloc((void**)&g->coords, sizeof(float3)*natm); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(g->coords, m->coords, sizeof(float3)*natm, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	// ALMOs
	cudaError = cudaMalloc((void**)&g->almos, sizeof(short4)*norb); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(g->almos, m->ALMOs, sizeof(short4)*norb, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);


	// density
	cudaError = cudaMalloc((void**)&g->qube, sizeof(float)*m->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&g->dm, sizeof(float)*norb*norb); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(g->dm, m->dm, sizeof(float)*norb*norb, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	// rpts probability fields
	cudaError = cudaMalloc((void**)&g->VNe000, sizeof(float)*m->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&g->VNe00z, sizeof(float)*m->npts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&g->VNe0yz, sizeof(float)*m->grid.x*m->grid.y); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&g->VNexyz, sizeof(float)*m->grid.x); assert(cudaError == cudaSuccess);

	// rpts space
	cudaError = cudaMalloc((void**)&g->rpts, sizeof(float3) * nrpts); assert(cudaError == cudaSuccess);

	// acsfs
	cudaError = cudaMalloc((void**)&g->acsf, sizeof(float) * nacsf); assert(cudaError == cudaSuccess);	


	*/
 	return g;
 }

 void scsf_gpu_free(SCSFGPU* g) {

 	cudaFree(g->VNe000);
 	cudaFree(g->VNe00z);
 	cudaFree(g->VNe0yz);
 	cudaFree(g->VNexyz);
 	cudaFree(g->Zs);
 	cudaFree(g->coords);
 	cudaFree(g->qube);
 	cudaFree(g->rpts);
 	cudaFree(g->acsf);

 	free(g);
 }


void scsf_compute(QMTools *obj, Molecule *mol) {


	SCSFGPU *g = scsf_gpu_allocate(obj, mol);


	// compute the random points
	//scsf_getpoints(obj, mol, g);

	// compute acsfs
	//acsf_compute(obj, mol, g);

	// compute density at the points
	//qube(obj, mol, g);

	/*
	random_setup(mol);
	acsf_allocate();


	// compute the positions of evaluation points
	random_grid(mol);
	// compute the cube
	qube(mol);
	// compute acsfs
	acsf_compute(mol);

	//cube_print_unwrap_ongpu(mol, mol->d_VNe, "molecule_29766_vne.bin");
	//cube_print_unwrap_ongpu(mol, mol->d_VNe_az, "molecule_29766_vne_az.bin");
	//cube_print_unwrap2d_ongpu(mol, mol->d_VNe_az_ay, "molecule_29766_vne_az_ay.bin");
	

	// write the complete output

	sprintf(moldir, "molecule_%i_output.bin", cID);
	FILE *fbin = fopen(moldir, "wb");
	
	// write the molecule
	molecule_write_bin(mol, fbin);

	// write the random evaluation points
	fwrite(&nrpts, sizeof(int), 1, fbin);
	fwrite(h_rpts, sizeof(float3), nrpts, fbin);

	// write the acsfs
	fwrite(&nacsf, sizeof(int), 1, fbin);
	fwrite(h_acsf, sizeof(float), nacsf, fbin);

	// write the correct density in those points
	fwrite(h_qube, sizeof(float), nrpts, fbin);

	fclose(fbin);

	molecule_free_complete(mol);

	random_free();
	acsf_free();
	*/


	scsf_gpu_free(g);
}