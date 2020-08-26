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



int main(int argc, char **argv) {

	// CREATE THE CONVOLVER AND LOAD REFERENCE MOLECULES
  
	Convolver cnv;

	cnv.nrefs = 1;
	cnv.refs = (Cube*)calloc(sizeof(Cube), cnv.nrefs);


	// LOAD ALL REF CUBES
	
	cube_load_reference(cnv.refs, "reference/qcube_CID_molecule_99173.bin");
	//cube_load_reference_dummy(cnv.refs);

	// ------------------


	// setup the GA
	cnv.populationSize = 256;
	cnv.mutationRate = 0.01f;
	cnv.lambda = 0.6f;

	// setup the convolver - this has to be done after popsize and reading refcubes
	convolver_setup(&cnv);


	// WRAP THE REF CUBES ******************************
	// this requires the thing to be setup cos itz done on the GPU
	for (int i = 0; i < cnv.nrefs; ++i) {
		
		cpu_cube_loadref(&cnv, cnv.refs+i); // this is not tested at all

		#ifdef DEBUGPRINT

		char fname[64]; sprintf(fname, "QREF_%05i.bin", i);
		cube_debug_print(cnv.refs+i, cnv.d_Q, fname);

		#endif
	}
	printf("reference cubes wrapped.\n");
	
	// *************************************************



	// start the GA run
	convolver_population_init(&cnv);
	ga_select_test(&cnv);

	for(int gen=0; gen<1000; gen++){
		printf("starting generation %05i...\n", gen);
		
		convolver_evaluate_population(&cnv);

		// TODO: write a restart/output

		convolver_evolve(&cnv);

	}
	






	return 0;


	// test the run on a cube
	convolver_reset(&cnv, cnv.refs); // clear grids
	convolver_makeP(&cnv, cnv.refs); // initialize
	// seems these do work as advertised

	cpu_Q_sum(&cnv, cnv.refs);


	// try to evolve the field
	number params[DNASIZE];
	params[PARAM_A0_DIFF] = 0.2f;
	params[PARAM_A0_LOS1] = 0.30f;
	params[PARAM_A0_LOS2] = 0.10f;
	params[PARAM_A0_LOS3] = 0.01f;
	params[PARAM_A0_AGEN] = 1.00f;
	params[PARAM_QQ_DIFF] = 0.10f;
	params[PARAM_QQ_TRNS] = 0.20f;
	cudaMemcpyToSymbol(c_parameters, params, sizeof(number) * DNASIZE, 0, cudaMemcpyHostToDevice);

	cpu_A0_propagate(&cnv, cnv.refs);
	cube_debug_print(cnv.refs, cnv.d_A0, "A0_0.bin");
	int converged = 0;
	int repo = 0;
	while(converged == 0 && repo < 10000) {
		cpu_A0_propagate(&cnv, cnv.refs);
		converged = cpu_Q_propagate(&cnv, cnv.refs);
		if(repo % 5 == 0)
			cpu_Q_sum(&cnv, cnv.refs);
		repo++;
	}


	printf("repos: %i\n", repo);
	cube_debug_print(cnv.refs, cnv.d_Q, "Q_1.bin");
	cube_debug_print(cnv.refs, cnv.d_A0, "A0_1.bin");


	/*
	cube_debug_print(cnv.refs, cnv.d_P, "P.bin");
	cube_debug_print(cnv.refs, cnv.d_Q, "Q.bin");
	cube_debug_print(cnv.refs, cnv.d_PmQ, "PmQ.bin");
	*/

	convolver_clear(&cnv);

	printf("all done!\n");
	return 0;
}



/*
void element_setup(Element *element) {
		
  double pscale = 1.0;
  double r;
	
	for(int i=0;i<element->dnaSize;i++) {
		r = ga_rand(element->engine);
		element->dna[i] = (2*r-1)*pscale;
	}
	
}

void element_mutate_custom(Element *element, int amount) {
  
  double r;
  int idx;

  for(int i=0; i<amount; i++) {
    idx = ga_irand(element->engine, element->dnaSize);
    r = ga_rand(element->engine);
    element->dna[idx] += (2*r-1);
  }
  
}



void element_evaluation(GA *engine, Element *element) {
	
  element->fitness = 0;
	double x;
	double y, y0;
	double error = 0;
	
	for(int i=0; i<=100; i++) {
	
	
		x = 2*PI*i*0.01;
		y = 0;
		for(int c=0;c<engine->dnaSize;c++) {
			y += element->dna[c] * cos(c*x);
		}
		
		y0 = 0.1 + 0.4*cos(x) - 0.3*cos(2*x) - 0.1*cos(3*x);
		error += fabs(y-y0);
	}
	
  element->fitness = error;
  
	double reg = 0;
	for(int i=0;i<engine->dnaSize; i++) reg += fabs(element->dna[i]);
	reg /= engine->dnaSize;
	
	//printf("element[%li] error=%lf  reg=%lf\n",element->ID, element->fitness, reg);
	element->fitness += reg*0.001;
	element->fitness *= -1;
}


int main()
{

	// *** GA ENGINE *** ***********************************************

	GA *engine = ga_new();
	engine->element_setup = element_setup;
	engine->element_mutate = element_mutate_custom;
	engine->element_evaluation = element_evaluation;

	ga_readsettings(engine, "ga.in");
	ga_setoutput(engine, "test.out");

	engine->output.freqRst = 1;
	
	// *****************************************************************

	/*
	int *rnds = calloc(64,sizeof(int));
	#pragma omp parallel
	{
		
		#pragma omp for
		for(int i=0;i<64; i++) {
			rnds[i] = ga_irand(engine, 10);
			printf("r[%i] = %i -- %i\n",i,rnds[i], omp_get_thread_num());
		}
	}
	
	//printf("r[%i] = %lf %lf \n",0,rnds[0],rnds[1]);
	
	//for(int i=0;i<20; i++)
		//printf("r[%i] = %i\n",i,rnds[i]);
	
	free(rnds);
	return 0;
	* /
	
	ga_init(engine, 10);
	/*
	for(int i=0; i<engine->populationSize; i++) {
	 for(int j=0; j<engine->dnaSize; j++) {
		 printf("%+.2lf ",engine->population[i].dna[j]);
	 }
	 printf("\n");
	}
	return 0;
	* /
	ga_evaluate(engine);
	
	for(int gen=0; gen<1000; gen++) {
		ga_step(engine);
		ga_evaluate(engine);
	}
	
	
	
	ga_free(engine);
	
	return 0;
}
*/





