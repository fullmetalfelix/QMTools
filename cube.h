

#ifndef CUBE
#define CUBE

#include "convolve.h"

/// The Cube represents one molecule.
typedef struct Cube Cube;
struct Cube {

	/// Molecule definition
	Molecule molecule;

	/// Origin of the cube
	float3 grid0;

	/// Shape of the cube
	dim3 gridSize;

	/// Shape of the GPU grid for this cube
	dim3 gpu_grid;

	/// Number of voxels in a cube.
	unsigned int npts;

	number *Q;
	number *d_Q;
};



void cube_load_reference(Cube *cube, const char *filename);
void cube_load_reference_dummy(Cube *cube);
void cube_debug_print(Cube *ref, number *gpusrc, const char *filename);




// CONSTANT MEMORY GLOBALS

/// Total number of points in a cube grid.
extern __constant__ unsigned int c_nCubes;

/// Total number of points in a cube grid.
extern __constant__ unsigned int c_cubeNpts;


/// Number of points along one side of the cube.
extern __constant__ int c_cubeSide;


/// Total number of GPU blocks assigned to one cube.
extern __constant__ int c_blocksPerCube;

/// Number of GPU blocks long one side of the cube.
extern __constant__ int c_blocksPerSide;


/// Diffusion constant of the A0 field.
extern __constant__ number c_A0_diff;

/// A0 generation factor.
extern __constant__ number c_A0_gen;

/// A0 spatial loss factor
extern __constant__ number c_A0_loss[3];


extern __constant__ uint3 c_cubeSideF;
extern __constant__ uint3 c_blocksPerCubeF;
extern __constant__ uint3 c_blocksPerSideF;




// functions

void cube_setup(Cube* cube, int nc, int c);
void cube_params(Cube* cube);
void cube_allocate(Cube* cube);
void cube_populate(Cube* cube);
void cube_free(Cube* cube);


#endif

