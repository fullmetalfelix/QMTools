
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
	cnv->offspring  = (Element*)malloc(sizeof(Element) * cnv->populationSize);
	cnv->dna = (number*)malloc(sizeof(number) * cnv->populationSize * DNASIZE);
	cnv->dna2= (number*)malloc(sizeof(number) * cnv->populationSize * DNASIZE);
	cnv->cdf = (number*)malloc(sizeof(number) * cnv->populationSize);
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
	cudaError = cudaMalloc((void**)&cnv->d_Ve, sizeof(number) * cnv->maxpts); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_partials, sizeof(number) * cnv->maxgrd); assert(cudaError == cudaSuccess);
	cudaError = cudaMalloc((void**)&cnv->d_deltaQmax, sizeof(uint)); assert(cudaError == cudaSuccess);

	uint zero = 0;
	cudaMemcpy(cnv->d_deltaQmax, &zero, sizeof(uint), cudaMemcpyHostToDevice);
}


void convolver_clear(Convolver *cnv) {

	free(cnv->population); free(cnv->offspring);
	free(cnv->dna);
	free(cnv->dna2);
	free(cnv->cdf);
	

	cudaFree(cnv->d_dna);
	//cudaFree(cnv->d_fitness);
	cudaFree(cnv->d_fitblok);

	cudaFree(cnv->d_coords);
	cudaFree(cnv->d_Zs);

	cudaFree(cnv->d_P);
	cudaFree(cnv->d_Q); cudaFree(cnv->d_Qn); cudaFree(cnv->d_Qref);
	cudaFree(cnv->d_PmQ);
	cudaFree(cnv->d_Ve);
	cudaFree(cnv->d_A0);cudaFree(cnv->d_A0n);
	cudaFree(cnv->d_partials);
	cudaFree(cnv->d_deltaQmax);
}






void convolver_mutation_0(Element *e, int mID) { // PARAM_A0_DIFF

	e->dna[mID] = 0.9f * (number)rand()/(number)(RAND_MAX);
}

void convolver_mutation_123(Element *e, int mID) { // PARAM_A0_LOS123
	
	e->dna[mID] = 1 * (number)rand()/(number)(RAND_MAX);
	//e->dna[mID] = 0.3f;
}

void convolver_mutation_4(Element *e, int mID) { // PARAM_A0_AGEN

	e->dna[mID] = 2 * (number)rand()/(number)(RAND_MAX);
	//e->dna[mID] = 1.0f;
}

void convolver_mutation_56(Element *e, int mID) { // PARAM_QQ_DIFF/TRNS

	e->dna[mID] = 0.5f * (number)rand()/(number)(RAND_MAX);
}

void (*mutationFunctions[DNASIZE])(Element *e, int mID) = {
	convolver_mutation_0, 
	convolver_mutation_123,
	convolver_mutation_123,
	convolver_mutation_123,
	convolver_mutation_4,
	convolver_mutation_56,
	convolver_mutation_56
};


void convolver_element_random(Element *e) {

	for(int i=0; i<DNASIZE; i++)
		(*mutationFunctions[i])(e, i);
}


void convolver_population_init(Convolver *cnv) {

	// setup the cumulative distro function for selection
	number *pdf = (number*)malloc(sizeof(number)*cnv->populationSize);
	number *cdf = cnv->cdf;

	number area = 0;

	for(int i=0; i<cnv->populationSize; i++) {
		pdf[i] = exp(-cnv->lambda * (number)i / cnv->populationSize);
		area += pdf[i];
	}
	
	for(int i=0; i<cnv->populationSize; i++) {
		pdf[i] /= area;
		cdf[i] = pdf[i];
	}
	
	for(int i=1; i<cnv->populationSize; i++)
		cdf[i] += cdf[i-1];

	free(pdf);
	// ---------------------------------------------------

	// initialize a population
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

	Cube *cube = &cnv->refs[cID];

	// set the grids
	convolver_reset(cnv, cube); // clear grids
	convolver_makeP(cnv, cube); // initialize
	convolver_makeVNN(cnv, cube);

	//cpu_Q_sum(cnv, cube); // normalize the total charge -- is it necessary?

	
	number deltaq, conv;
	int repo = 0;
	while(conv >= cnv->dqTolerance && repo < cube->maxside * 100) {

		for(int i=0; i<10; ++i)
			cpu_Vee_propagate(cnv, cube);

		deltaq = cpu_Q_propagate(cnv, cube, (repo % 10 == 0));
		if(repo % 10 == 0) conv = deltaq;

		repo++;
	}

	// compute fitness on this cube
	number mismatch = cpu_Q_diff(cnv, cube);
	#ifdef DEBUGPRINT

	sprintf(fname, "Q_final_%05i_%05i.bin", mID,cID);
	cube_debug_print(cnv, cube, cnv->d_Q, fname);

	printf("mismatch %e\n",mismatch);
	assert(1 == 0);
	#endif
	// accumulate
	cnv->population[mID].fitness -= mismatch;
}

void convolver_evaluate_cube_alt(Convolver *cnv, int cID, int mID) {

	Cube *cube = &cnv->refs[cID];

	#ifdef DEBUGPRINT
	char fname[64];
	printf("evaluating on ref CUBE %i... [%i %i %i]\n", cID, cube->gridSize.x, cube->gridSize.y, cube->gridSize.z);
	#endif


	// set the grids
	convolver_reset(cnv, cube); // clear grids
	convolver_makeP(cnv, cube); // initialize

	// copy the correct Q into GPU and update the PmQ
	convolver_makePmQ(cnv, cube);

	#ifdef DEBUGPRINT

	sprintf(fname, "Q_%05i_%05i.bin",mID,cID);
	cube_debug_print(cnv,cube, cnv->d_Q, fname);

	#endif

	// converge the A0 field
	int converged = 0;
	int repo = 0;
	number dA0;
	/*while(converged == 0 || repo < cube->maxside) {
		dA0 = cpu_A0_propagate_tally(cnv, cube);
		converged = dA0 < 0.0391f;
		repo++;
		if(repo == cube->maxside * 10) {
			printf("A0 iteration limit\n");
			break;
		}
	}
	*/

	for(repo=0; repo < 50000; repo++) {

		//sprintf(fname, "A0_%i_%05i_%05i.bin", repo,mID,cID);
		//cube_debug_print(cnv->refs, cnv->d_A0, fname);

		cpu_A0_propagate(cnv, cube);
		
		if(repo % 10000 == 0) {
			dA0 = cpu_A0_propagate_tally(cnv, cube);
			printf("dA0 = %15.8e\n", dA0);
			//sprintf(fname, "A0_%i_%05i_%05i.bin", mID,cID,repo/10000);
			//cube_debug_print(cube, cnv->d_A0, fname);
		}
	}
	dA0 = cpu_A0_propagate_tally(cnv, cube);
	printf("dA0 last = %15.8e\n", dA0);
}


void convolver_evaluate_model(Convolver *cnv, int mID) {

	#ifdef DEBUGPRINT
	printf("evaluating ELEMENT %i...\n", mID);
	#endif

	// load the model parameters to gpu cmem
	cudaMemcpyToSymbol(c_parameters, cnv->population[mID].dna, sizeof(number) * DNASIZE, 0, cudaMemcpyHostToDevice);

	printf("DNA: ");
	for (int i = 0; i < DNASIZE; ++i) {
		printf("%f\t", cnv->population[mID].dna[i]);
	}
	printf("\n");

	// reset fitness
	cnv->population[mID].fitness = 0;

	// loop over all the ref cubes
	for(int cID=0; cID<cnv->nrefs; cID++) {

		if(cnv->alt == 0) convolver_evaluate_cube(cnv, cID, mID);
		else convolver_evaluate_cube_alt(cnv, cID, mID);

	}
}


int ga_compare_fitness (const void *aa, const void *bb) {

  Element *a = (Element*)aa;
  Element *b = (Element*)bb;

  if(b->fitness > a->fitness) return 1;
  else if (b->fitness < a->fitness) return -1;
  else return 0;
}


void convolver_checkpoint_write(Convolver *cnv) {

	FILE *fout = fopen("checkpoint.rst", "wb");

	fwrite(&cnv->populationSize, sizeof(int), 1, fout);

	for(int i=0; i<cnv->populationSize; i++) {
		fwrite(cnv->population[i].dna, sizeof(number), DNASIZE, fout);
		fwrite(&cnv->population[i].fitness, sizeof(number), 1, fout);
	}

	fclose(fout);
}

void convolver_checkpoint_read(Convolver *cnv, const char* filename) {

	FILE *fout = fopen(filename, "rb");

	int psize; fread(&cnv->populationSize, sizeof(int), 1, fout);
	if(psize > cnv->populationSize) {
		psize = cnv->populationSize;
		printf("WARNING: not all elements in the checkpoint will be read\n");
	} else if(psize < cnv->populationSize) {

		printf("WARNING: population is larger than the one in checkpoint\n");
	}

	for(int i=0; i<psize; i++) {
		fread(cnv->population[i].dna, sizeof(number), DNASIZE, fout);
		fread(&cnv->population[i].fitness, sizeof(number), 1, fout);
	}

	fclose(fout);
}

void convolver_evaluate_population(Convolver *cnv) {


	for(int mID=0; mID<cnv->populationSize; mID++) {
		convolver_evaluate_model(cnv, mID);
	}

	// sort the elements by fitness... sorting of structs by fitness field
	qsort(cnv->population, cnv->populationSize, sizeof(Element), ga_compare_fitness);

	// write restart file
	convolver_checkpoint_write(cnv);
}



int ga_select(Convolver *cnv, int except);
int ga_select(Convolver *cnv, int except) {

	int picked = except;
	number r = (number)rand()/(number)(RAND_MAX);
	for (int i=0; i<cnv->populationSize; i++) {
		if (cnv->cdf[i] >= r) {
			picked = i;
			break;
		}
	}

	if(picked == except) {
		picked = (picked ==  cnv->populationSize-1)? picked - 1 : picked + 1;
	}

	return picked;
}


void ga_select_test(Convolver *cnv) {

	printf("selection test...\n");
	FILE *fout = fopen("select.test", "w");
	for(int i=0; i<100000; i++)
		fprintf(fout, "%i\n", ga_select(cnv, -1));

	fclose(fout);
}






/// Create the offspring population and switch the pointers
void convolver_evolve(Convolver *cnv) {

	printf("evolving...\n");
	number r;

	for(int i=0; i<cnv->populationSize; i++) {

		printf("selecting %05i \n", i);
		Element *o = &cnv->offspring[i];
		o->dna = &cnv->dna2[DNASIZE * i];
		o->fitness = 0;

		if(i<cnv->keepers) {
			// keep the element... copy dna
			memcpy(o->dna, cnv->population[i].dna, sizeof(number) * DNASIZE);

		} else if(i >= cnv->populationSize - cnv->tossers) {
			// completely rerandomize
			convolver_element_random(o);

		} else {
			// normal mix & mutate

			// select parents
			int p1 = ga_select(cnv, -1); printf("p1 %i\n", p1);
			int p2 = ga_select(cnv, p1); printf("p2 %i\n", p2);

			// mix
			for(int j=0; j<DNASIZE; j++) {
				// take value from either parent
				r = (number)rand()/(number)(RAND_MAX);
				o->dna[j] = (r > 0.5f)? cnv->population[p1].dna[j] : cnv->population[p2].dna[j];
			}

			// mutate
			r = (number)rand()/(number)(RAND_MAX);
			if(r < cnv->mutationRate) {
				r = DNASIZE * ((number)rand()/RAND_MAX);
				int mID = (int)floor(r);
				mID = mID % DNASIZE;
				(*mutationFunctions[mID])(o, mID);
			}
		}
	}

	// now we have the new population in offspring
	// and the new dna in dna2

	// switch the dna pointers
	number *tmp = cnv->dna;
	cnv->dna = cnv->dna2;
	cnv->dna2= tmp;

	// switch the population pointers
	Element *tmpe = cnv->population;
	cnv->population = cnv->offspring;
	cnv->offspring = tmpe;
}




void asd(Convolver *cnv, int cID, int mID) {

	char fname[128];
	Cube *cube = &cnv->refs[cID];

	// load the model parameters to gpu cmem
	cudaMemcpyToSymbol(c_parameters, cnv->population[mID].dna, sizeof(number) * DNASIZE, 0, cudaMemcpyHostToDevice);


	// set the grids
	convolver_reset(cnv, cube); // clear grids
	convolver_makeP(cnv, cube); // initialize P and Q
	convolver_makeVNN(cnv, cube);


	// copy the solution
	//cudaMemcpy(cnv->d_Q, cube->Q, sizeof(number) * cube->npts, cudaMemcpyHostToDevice);

	cpu_Q_sum(cnv, cube); // normalize the total charge -- is it necessary?

	// copy the parameters to gpu constant memory -- no! the parameters should already be there!
	
	// generate the initial field
	cube_debug_print(cnv, cube, cnv->d_A0, "VNN.bin");


	number converged = 0;
	int repo = 0;
	int printN = 1000;

	// first relax the Vee using the correct solution

	// then try to evolve the density... train so that there will be no more diffusion


	// try the full thing to see where it goes

	while(repo < 10*printN) {
		
		for(int k=0;k<50;k++)
			cpu_Vee_propagate(cnv, cube);
		
		converged = cpu_Q_propagate(cnv, cube, 0);

		if(repo % 10 == 0)
			cpu_Q_sum(cnv, cube);
		
		
		if(repo % printN == 0) {
			printf("%15.8e\n", converged);
			sprintf(fname, "Q_%05i_%05i_%05i.bin", repo/printN,mID,cID);
			cube_debug_print(cnv, cube, cnv->d_Q, fname);
			sprintf(fname, "V_%05i_%05i_%05i.bin", repo/printN,mID,cID);
			cube_debug_print(cnv, cube, cnv->d_Ve, fname);
		}

		repo++;
	}

	/*
	repo = 0;
	printN = 1;
	while(repo < 10*printN) {

		cpu_Vee_propagate(cnv, cube);
		converged = cpu_Q_propagate(cnv, cube);

		if(repo % 10 == 0)
			cpu_Q_sum(cnv, cube);

		if(repo % printN == 0) {
			printf("%15.8e\n", converged);
			sprintf(fname, "Q_%05i_%05i_%05i.bin", repo/printN,mID,cID);
			cube_debug_print(cnv, cube, cnv->d_Q, fname);
			sprintf(fname, "V_%05i_%05i_%05i.bin", repo/printN,mID,cID);
			cube_debug_print(cnv, cube, cnv->d_Ve, fname);
		}

		repo++;
	}
	*/
}






