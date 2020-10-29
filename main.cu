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
#include <unistd.h>


int main(int argc, char **argv) {

	// CREATE THE CONVOLVER AND LOAD REFERENCE MOLECULES
  
	Convolver cnv;

	cnv.alt = 0;
	cnv.nrefs = 3;
	cnv.refs = (Cube*)calloc(sizeof(Cube), cnv.nrefs);
	
	// LOAD ALL REF CUBES
	
	cube_load_reference(&cnv.refs[0], "reference/qcube_CID_molecule_7918.bin");
	cube_load_reference(&cnv.refs[1], "reference/qcube_CID_molecule_85689.bin");
	cube_load_reference(&cnv.refs[2], "reference/qcube_CID_molecule_99173.bin");
	//cube_load_reference_dummy(cnv.refs);
	
	// ------------------
	
	// setup the GA
	cnv.populationSize = 64;
	cnv.mutationRate = 0.01f;
	cnv.lambda = 0.6f;
	cnv.keepers = 0;
	cnv.tossers = 0;

	// read command line options
	int opt;
	char *restartfile = NULL;

	// put ':' in the starting of the 
	// string so that program can  
	//distinguish between '?' and ':'  
	while((opt = getopt(argc, argv, ":p:l:m:k:t:hR:")) != -1) {  
		switch(opt) {  
			case 'p':
				cnv.populationSize = atoi(optarg);
				break;
			case 'k':
				cnv.keepers = atoi(optarg);
				break;
			case 't':
				cnv.tossers = atoi(optarg);
				break;
			case 'l':
				cnv.lambda = (number)atof(optarg);
				break;
			case 'm':
				cnv.mutationRate = (number)atof(optarg);
				break;
			case 'R':
				restartfile = optarg;
			case ':':
				printf("option needs a value\n");
				break;
			case '?':
				printf("unknown option: %c\n", optopt);
				break;
			case 'h':
				printf("Command-line options:\n");
				printf("-p [int]\tpopulation size\n");
				printf("-k [int]\tkeepers\n");
				printf("-t [int]\ttossers\n");
				printf("-l [float]\tselection lambda\n");
				printf("-m [float]\tmutation rate\n");
				return 0;
				break;
		}
	}


	// setup the convolver - this has to be done after popsize and reading refcubes
	convolver_setup(&cnv);


	// WRAP THE REF CUBES ******************************
	// this requires the thing to be setup cos itz done on the GPU
	for (int i=0; i<cnv.nrefs; ++i) {
		
		cpu_cube_loadref(&cnv, &cnv.refs[i]); // this is not tested at all

		#ifdef DEBUGPRINT

		char fname[64]; sprintf(fname, "QREF_%05i.bin", i);
		cube_debug_print(&cnv, &cnv.refs[i], cnv.d_Q, fname);

		#endif
	}
	printf("reference cubes wrapped.\n");
	
	// *************************************************



	// start the GA run
	convolver_population_init(&cnv);
	if(restartfile != NULL) {
		printf("restarting from: %s\n", restartfile);
		convolver_checkpoint_read(&cnv, restartfile);	
	}

	ga_select_test(&cnv);

	FILE *flog = fopen("ga.log", "w");

	for(int gen=0; gen<1000; gen++) {
		printf("starting generation %05i...\n", gen);
		
		convolver_evaluate_population(&cnv); // this also writes the checkpoint
		
		// write the log
		for(int c=0; c<DNASIZE; c++)
			fprintf(flog, "%f\t", cnv.population[0].dna[c]);
		fprintf(flog, "%f\n", cnv.population[0].fitness);
		
		convolver_evolve(&cnv);
	}
	
	fclose(flog);


	convolver_clear(&cnv);

	printf("all done!\n");
	return 0;
}


