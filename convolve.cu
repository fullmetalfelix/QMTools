
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cube.h"
#include "convolve.h"
#include <assert.h>




/*
This model uses:
Q: cube of electronic charges - mutable (positive values)
P: cube of nuclear charegs - fixed (positive values)
PmQ: cube of both nuclear and electronic charges
A0: cube with "potential" field

*/

/* OUTLINE OF THE CODE:
	
	# read input file

	the cube object contains all information of all cubes


	evolution of a system:
		1. update Q using Q, A0, ...
		2. propagate field



	GA run to fit parameters:

	load some reference cubes from CCSD molecules


	loop over population elements e:
		
		load GA parameters of e into constant memory
		evaluate fitness of e:
			
			loop over reference cubes r:
				setup the P tensor corresponding to r (on cpu and copy)
				setup the starting Q and PmQ tensors
				zero the A0 tensor
				evolve Q until... ?converged?
			
			compare Q to reference r -> MRE?

			asd


		asd



	MRE allocation:
		(GPU) one value for each thread block, and each reference
*/


/// Allocates all necessary stuff on CPU and GPU.
/// Also the GA population 
void convolver_setup(Convolver *cnv) {

	cudaError_t cudaError;

	cnv->gpu_block = dim3(8,8,8);

	cnv->population = (Element*)malloc(sizeof(Element) * cnv->populationSize);
	cnv->dna = (number*)malloc(sizeof(number) * cnv->populationSize * DNASIZE);
	cnv->dna2= (number*)malloc(sizeof(number) * cnv->populationSize * DNASIZE);
	//cnv->fitness = (number*)calloc(sizeof(number), cnv->populationSize);
	cnv->dqTolerance = 1.0e-32;


	cudaError = cudaMalloc((void**)&cnv->d_dna, sizeof(number) * cnv->populationSize * DNASIZE); assert(cudaError == cudaSuccess);
	//cudaError = cudaMalloc((void**)&cnv->d_fitness, sizeof(number) * cnv->populationSize); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_fitblok, sizeof(number) * MAXBLOCKS); assert(cudaError == cudaSuccess);


	// atomic position temp allocations
	cudaError = cudaMalloc((void**)&cnv->d_coords, sizeof(float3) * MAXATOMS); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_Zs, sizeof(int) * MAXATOMS); assert(cudaError == cudaSuccess);


	// compute the max size of the ref grid
	cnv->maxpts = 0;
	for (int i = 0; i < cnv->nrefs; ++i)
		cnv->maxpts = (cnv->refs[i].npts > cnv->maxpts)? cnv->refs[i].npts : cnv->maxpts;
	cnv->maxgrd = cnv->maxpts / 512;
	printf("max refgrid points: %i\n", cnv->maxpts);
	printf("max refgrid blocks: %i\n", cnv->maxgrd);


	// allocate space for the calculations
	cudaError = cudaMalloc((void**)&cnv->d_P, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_Q, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_Qn, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_Qref, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_PmQ, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_A0, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_A0n, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_partials, sizeof(number) * cnv->maxgrd); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_deltaQmax, sizeof(uint)); assert(cudaError == cudaSuccess);

	uint zero = 0;
	cudaMemcpy(cnv->d_deltaQmax, &zero, sizeof(uint), cudaMemcpyHostToDevice);
}


void convolver_clear(Convolver *cnv) {

	free(cnv->population);
	free(cnv->dna);
	free(cnv->dna2);
	//free(cnv->fitness);

	cudaFree(cnv->d_dna);
	//cudaFree(cnv->d_fitness);
	cudaFree(cnv->d_fitblok);

	cudaFree(cnv->d_coords);
	cudaFree(cnv->d_Zs);

	cudaFree(cnv->d_P);
	cudaFree(cnv->d_Q); cudaFree(cnv->d_Qn); cudaFree(cnv->d_Qref);
	cudaFree(cnv->d_PmQ);
	cudaFree(cnv->d_A0);cudaFree(cnv->d_A0n);
	cudaFree(cnv->d_partials);
	cudaFree(cnv->d_deltaQmax);
}



void convolver_element_random(Element *e) {

	e->dna[PARAM_A0_DIFF] = 1 * (float)rand()/(float)(RAND_MAX);
	e->dna[PARAM_A0_AGEN] = 1 * ((float)rand()/(float)(RAND_MAX));

	// good limit for these ones?
	e->dna[PARAM_A0_LOS1] = 1 * ((float)rand()/(float)(RAND_MAX));
	e->dna[PARAM_A0_LOS2] = 1 * ((float)rand()/(float)(RAND_MAX));
	e->dna[PARAM_A0_LOS3] = 1 * ((float)rand()/(float)(RAND_MAX));

	// magnitude of gen/los?
	number losmag = e->dna[PARAM_A0_LOS1] + e->dna[PARAM_A0_LOS2] + e->dna[PARAM_A0_LOS3];
	printf("magn gen/loss: %f/%f\n", e->dna[PARAM_A0_AGEN], losmag);


	e->dna[PARAM_QQ_DIFF] = 0.5f * (float)rand()/(float)(RAND_MAX);
	e->dna[PARAM_QQ_TRNS] = 0.5f * (float)rand()/(float)(RAND_MAX);
}


void convolver_population_init(Convolver *cnv) {

	for (int i=0; i<cnv->populationSize; ++i) {
		cnv->population[i].dna = cnv->dna + DNASIZE*i;
		convolver_element_random(cnv->population + i);

		#ifdef DEBUGPRINT
		printf("element %5i: ", i);
		for (int j = 0; j < DNASIZE; ++j) {
			printf("%8e\t",cnv->population[i].dna[j]);
		}
		printf("\n");
		#endif
	}
	cudaMemcpy(cnv->d_dna, cnv->dna, sizeof(number) * DNASIZE * cnv->populationSize, cudaMemcpyHostToDevice);
}



/// Runs the currently loaded model on a reference cube
void convolver_evaluate_cube(Convolver *cnv, int cID, int mID) {

	
	#ifdef DEBUGPRINT
	char fname[64];
	printf("evaluating on ref cube %05i...\n", cID);
	#endif

	Cube *cube = cnv->refs + cID;

	// set the grids
	convolver_reset(cnv, cube); // clear grids
	convolver_makeP(cnv, cube); // initialize
	cpu_Q_sum(cnv, cube); // normalize the total charge -- is it necessary?

	// copy the parameters to gpu constant memory -- no! the parameters should already be there!
	
	// generate the initial field
	cpu_A0_propagate(cnv, cube);
	//cube_debug_print(cube, cnv->d_A0, "A0_0.bin");

	int converged = 0;
	int repo = 0;
	while(converged == 0 && repo < MAXREPOS) {
		cpu_A0_propagate(cnv, cube);
		converged = cpu_Q_propagate(cnv, cube);

		if(repo % 1 == 0)
			cpu_Q_sum(cnv, cube);
		repo++;
	}


	#ifdef DEBUGPRINT
	sprintf(fname, "A0_%i_%05i_%05i.bin", repo,mID,cID);
	cube_debug_print(cnv->refs, cnv->d_A0, fname);
	sprintf(fname, "Q_%i_%05i_%05i.bin", repo,mID,cID);
	cube_debug_print(cnv->refs, cnv->d_Q, fname);
	#endif

	// compute fitness on this cube
	number mismatch = cpu_Q_diff(cnv, cube);
	#ifdef DEBUGPRINT
	printf("mismatch %e\n",mismatch);
	#endif
	// accumulate
	cnv->population[mID].fitness += mismatch;
}


void convolver_evaluate_model(Convolver *cnv, int mID) {

	#ifdef DEBUGPRINT
	printf("evaluating element %05i...\n", mID);
	#endif

	// load the model parameters to gpu cmem
	cudaMemcpyToSymbol(c_parameters, cnv->population[mID].dna, sizeof(number) * DNASIZE, 0, cudaMemcpyHostToDevice);


	// reset fitness
	cnv->population[mID].fitness = 0;

	// loop over all the refs
	for(int cID=0; cID<cnv->nrefs; cID++) {

		convolver_evaluate_cube(cnv, cID, mID);

	}
}


int ga_compare_fitness (const void *aa, const void *bb) {

  Element *a = (Element*)aa;
  Element *b = (Element*)bb;

  if(b->fitness > a->fitness) return 1;
  else if (b->fitness < a->fitness) return -1;
  else return 0;
}

void convolver_evaluate_population(Convolver *cnv) {


	for(int mID=0; mID<cnv->populationSize; mID++) {
		convolver_evaluate_model(cnv, mID);
	}

	// sort the elements by fitness... sorting of structs by fitness field
	qsort(cnv->population, cnv->populationSize, sizeof(Element), ga_compare_fitness);
	


}










