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



//Read BasisSet variables from .bin file

void basisset_ini(BasisSet *basisset, const char *filename) {

	int ncmax = 0;
	int maxorbs = 0;

	printf("reading basis set from: %s\n", filename);

	FILE *fbin = fopen(filename, "rb");
	int m; fread(&m, sizeof(int), 1, fbin);			// READ THE # of ATOMS IN THIS BASIS SET

	//Initialize variables
	basisset->atomOffset = (int*)malloc(sizeof(int) * 100);
	basisset->nshells = (int*)calloc(MAXAOS*100, sizeof(int));
	memset(basisset->atomOffset, -1, 100);
	

	basisset->shellOffset 	= (int*)calloc(MAXAOS*100, sizeof(int));
	basisset->Ls 			= (int*)calloc(MAXAOS*100, sizeof(int));

	basisset->alphas = (float*)calloc(MAXAOS*MAXAOC*100, sizeof(float));
	basisset->coeffs = (float*)calloc(MAXAOS*MAXAOC*100, sizeof(float));

	int offset = 0;
	
	basisset->global_m_values = (int**)malloc(sizeof(int*) * 3);
	basisset->global_m_values[0] = (int*)malloc(sizeof(int));
	basisset->global_m_values[0][0] = 0;
	basisset->global_m_values[1] = (int*)malloc(sizeof(int)*3);
	basisset->global_m_values[1][0] = 0;
	basisset->global_m_values[1][1] = 1;
	basisset->global_m_values[1][2] = -1;
	basisset->global_m_values[2] = (int*)malloc(sizeof(int)*5);
	basisset->global_m_values[2][0] = 0;
	basisset->global_m_values[2][1] = 1;
	basisset->global_m_values[2][2] = -1;
	basisset->global_m_values[2][3] = 2;
	basisset->global_m_values[2][4] = -2;
	

	int orbcount = 0;
	int z, tmpi, nc;
	double *buffer = (double*)malloc(sizeof(double) * MAXAOC);

	//Read variables
	for(int i=0; i<m; i++) {

		// read the Z of this atom
		fread(&z, sizeof(int), 1, fbin);
		basisset->atomOffset[z] = orbcount;

		// read the shells info
		fread(&tmpi, sizeof(int), 1, fbin); basisset->nshells[z] = tmpi;
		maxorbs = (tmpi>maxorbs)? tmpi : maxorbs;

		// maybe we want to keep parameters only for the ones we actually use?

		for(int j=0; j<basisset->nshells[z]; j++) {

			// read the L of this shell
			fread(&(basisset->Ls[orbcount]), sizeof(int), 1, fbin);
			// number of primitives in the contraction
			fread(&nc, sizeof(int), 1, fbin);
			
			basisset->shellOffset[orbcount] = offset;

			memset(buffer, 0, MAXAOC);
			fread(buffer, sizeof(double), nc, fbin);
			for(int k=0; k<nc; k++)
				basisset->alphas[offset+k] = (float)buffer[k];

			memset(buffer, 0, MAXAOC);
			fread(buffer, sizeof(double), nc, fbin);
			for(int k=0; k<nc; k++)
				basisset->coeffs[offset+k] = (float)buffer[k];

			//printf("B%i\tAO%i\n", i, j);
			//printf("Z=%03i AO%02i\tl=%i nc=%i\n",z, j, basisset->Ls[orbcount], nc);
			
			
			ncmax = (nc>ncmax)? nc : ncmax;

			orbcount++;
			offset += MAXAOC;
		}
	}

	basisset->nparams = offset;

	fclose(fbin);
	free(buffer);


	// basis set parameters to device memory
	cudaError_t cudaError;
	cudaError = cudaMalloc((void**)&basisset->d_alphas, sizeof(float)*basisset->nparams); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(basisset->d_alphas, basisset->alphas, sizeof(float)*basisset->nparams, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	cudaError = cudaMalloc((void**)&basisset->d_coeffs, sizeof(float)*basisset->nparams); assert(cudaError == cudaSuccess);
	cudaError = cudaMemcpy(basisset->d_coeffs, basisset->coeffs, sizeof(float)*basisset->nparams, cudaMemcpyHostToDevice); assert(cudaError == cudaSuccess);

	printf("basis set read - maxorbs=%i  ncmax=%i - nparams=%i\n", maxorbs, ncmax, offset);
}

void basisset_del(BasisSet *basisset) {

	free(basisset->atomOffset);
	free(basisset->nshells);

	free(basisset->shellOffset);
	free(basisset->Ls);
	
	free(basisset->alphas);
	free(basisset->coeffs);

	free(basisset->global_m_values[0]);
	free(basisset->global_m_values[1]);
	free(basisset->global_m_values[2]);
	free(basisset->global_m_values);

	cudaFree(basisset->d_alphas);
	cudaFree(basisset->d_coeffs);
}

/*
void scsf_basisset_load(SCSF *obj, const char *filename) {

	int ncmax = 0;
	int maxorbs = 0;

	printf("reading basis set from: %s\n", filename);

	FILE *fbin = fopen(filename, "rb");
	int m; fread(&m, sizeof(int), 1, fbin);			// READ THE # of ATOMS IN THIS BASIS SET

	BasisSet *basisset = (BasisSet*)malloc(sizeof(BasisSet));

	basisset->atomOffset = (int*)malloc(sizeof(int) * 100);
	basisset->nshells = (int*)calloc(MAXAOS*100, sizeof(int));
	memset(basisset->atomOffset, -1, 100);
	

	basisset->shellOffset 	= (int*)calloc(MAXAOS*100, sizeof(int));
	basisset->Ls 			= (int*)calloc(MAXAOS*100, sizeof(int));

	basisset->alphas = (float*)calloc(MAXAOS*MAXAOC*100, sizeof(float));
	basisset->coeffs = (float*)calloc(MAXAOS*MAXAOC*100, sizeof(float));

	int offset = 0;
	
	basisset->global_m_values = (int**)malloc(sizeof(int*) * 3);
	basisset->global_m_values[0] = (int*)malloc(sizeof(int));
	basisset->global_m_values[0][0] = 0;
	basisset->global_m_values[1] = (int*)malloc(sizeof(int)*3);
	basisset->global_m_values[1][0] = 0;
	basisset->global_m_values[1][1] = 1;
	basisset->global_m_values[1][2] = -1;
	basisset->global_m_values[2] = (int*)malloc(sizeof(int)*5);
	basisset->global_m_values[2][0] = 0;
	basisset->global_m_values[2][1] = 1;
	basisset->global_m_values[2][2] = -1;
	basisset->global_m_values[2][3] = 2;
	basisset->global_m_values[2][4] = -2;
	

	int orbcount = 0;
	int z, tmpi, nc;
	double *buffer = (double*)malloc(sizeof(double) * MAXAOC);


	for(int i=0; i<m; i++) {

		// read the Z of this atom
		fread(&z, sizeof(int), 1, fbin);
		basisset->atomOffset[z] = orbcount;

		// read the shells info
		fread(&tmpi, sizeof(int), 1, fbin); basisset->nshells[z] = tmpi;
		maxorbs = (tmpi>maxorbs)? tmpi : maxorbs;

		// maybe we want to keep parameters only for the ones we actually use?

		for(int j=0; j<basisset->nshells[z]; j++) {

			// read the L of this shell
			fread(&(basisset->Ls[orbcount]), sizeof(int), 1, fbin);
			// number of primitives in the contraction
			fread(&nc, sizeof(int), 1, fbin);
			
			basisset->shellOffset[orbcount] = offset;

			memset(buffer, 0, MAXAOC);
			fread(buffer, sizeof(double), nc, fbin);
			for(int k=0; k<nc; k++)
				basisset->alphas[offset+k] = (float)buffer[k];

			memset(buffer, 0, MAXAOC);
			fread(buffer, sizeof(double), nc, fbin);
			for(int k=0; k<nc; k++)
				basisset->coeffs[offset+k] = (float)buffer[k];

			//printf("B%i\tAO%i\n", i, j);
			//printf("Z=%03i AO%02i\tl=%i nc=%i\n",z, j, basisset->Ls[orbcount], nc);
			
			
			ncmax = (nc>ncmax)? nc : ncmax;

			orbcount++;
			offset += MAXAOC;
		}
	}

	basisset->nparams = offset;

	fclose(fbin);
	free(buffer);
	printf("basis set read - maxorbs=%i  ncmax=%i - nparams=%i\n", maxorbs, ncmax, offset);
	obj->basisset = basisset;
}


void scsf_basisset_free(SCSF *obj) {

	BasisSet *basisset = obj->basisset;

	free(basisset->atomOffset);
	free(basisset->nshells);

	free(basisset->shellOffset);
	free(basisset->Ls);
	
	free(basisset->alphas);
	free(basisset->coeffs);

	free(basisset->global_m_values[0]);
	free(basisset->global_m_values[1]);
	free(basisset->global_m_values[2]);
	free(basisset->global_m_values);

	free(basisset);
}
*/