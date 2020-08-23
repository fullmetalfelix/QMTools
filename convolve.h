
#ifndef CONVOLVE
#define CONVOLVE

typedef struct Cube Cube;

#define B 8
#define Bp 9
#define Bp1 (B+2)

#define B_2 (B*B)
#define B_3 (B*B*B)

#define Bp1_2 (Bp1 * Bp1)
#define Bp1_3 (Bp1 * Bp1 * Bp1)

#define LB 512


// GA population size
#define POPSIZE 128

// size of DNA for ONE element
#define DNASIZE 7

// number of reference molecules
#define NREFS 10

// DEBUG VALUE!
#define MAXREPOS 1000

// maximum number of blocks fitting in one grid (cube 512**3 with blocksize 8**3)
#define MAXBLOCKS 262144

// value of the grid spacing in BOHR
#define DS 0.18897261246257701551
#define OneOverSqrt2 0.7071067811865475
#define OneOverSqrt3 0.5773502691896258
#define OneOverDIFFTOT 0.05234482976098482

#define MAXATOMS 50

#define ANG2BOR 1.8897259886


// offset of parameters in constant memory
#define PARAM_A0_DIFF 0
#define PARAM_A0_LOS1 1
#define PARAM_A0_LOS2 2
#define PARAM_A0_LOS3 3
#define PARAM_A0_AGEN 4
#define PARAM_QQ_DIFF 5
#define PARAM_QQ_TRNS 6

typedef float number;

typedef struct Molecule Molecule;
struct Molecule {

	int natoms;
	int *Zs;
	float3 *coords; // in bohr, translated to that the 0 is at the origin of the grid

	int qtot;

};



/*
ASSUMPTIONS:

	CUBE DEFINITION:
	a cube is the information of one system, including charges and fields on their respective 3D grids.
	the code might operate on multiple systems.


	GPU KERNELS:
	one kernel call operates on one cube at a time, in a stream dedicated to that particular system.


	
	3D DATA CUBES:
	all grids have cubic shape, and the side is a multiple of the GPU block size.
	cubes are stored in a blockwise pattern
	for example (2D)

	 0  1  2  3 
	 4  5  6  7 
	 8  9 10 11
	12 13 14 15

	the first gpu 2D block would load 0 1 4 5
	Then the optimal storage pattern would be:

	0 1 4 5
	2 3 6 7
	...

	this way each block loads a contiguous chunk of memory
*/

typedef struct Element Element;
struct Element {

	number *dna;
	number fitness;
};


typedef struct Convolver Convolver;
struct Convolver {

	unsigned int populationSize;
	Element *population;
	number *dna, *d_dna;
	number *dna2;

	//number *fitness;
	//number *d_fitness;
	number *d_fitblok; // this stores the 

	number mutationRate;


	int nrefs;
	Cube *refs;

	/// Largest amount of grid points in all reference molecules.
	unsigned int maxpts;
	unsigned int maxgrd;

	float3 *d_coords;
	int *d_Zs;

	number *d_P;
	number *d_Q, *d_Qn;
	number *d_PmQ;
	number *d_A0, *d_A0n;
	number *d_partials;
	number *d_Qref;

	uint *d_deltaQmax;
	uint deltaQmax;
	float dqTolerance;

	dim3 gpu_block;
};





// GA functions
void convolver_population_init(Convolver *cnv);
void convolver_evaluate_population(Convolver *cnv);



// main object functions
void convolver_setup(Convolver *cnv);
void convolver_clear(Convolver *cnv);





/*
void cpu_cube_unwrap(Cube *cube, number *dst, number *src);
void cpu_A0_propagate(Cube *cube);
void cpu_A0_propagate_all(Cube *cube);
void cpu_Q_propagate(Cube *cube);
*/

// GPU wrappers functions
void convolver_reset(Convolver *cnv, Cube *cube);
void convolver_makeP(Convolver *cnv, Cube *cube);

void cpu_cube_loadref(Convolver *cnv, Cube *cube);
void cpu_A0_propagate(Convolver *cnv, Cube *cube);
int cpu_Q_propagate(Convolver *cnv, Cube *cube);
void cpu_Q_sum(Convolver *cnv, Cube *cube);


number cpu_Q_diff(Convolver *cnv, Cube *cube);




// GPU constant mem
extern __constant__ number c_parameters[DNASIZE];


#endif


