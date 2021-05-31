#ifndef KERNEL_QUBE
#define KERNEL_QUBE

#define SUBGRID 4
#define SUBGRIDDX 0.25f
#define SUBGRIDDX2 0.125f
#define SUBGRIDiV 0.015625f



/*
void qube(Molecule *mol);
void qube_compare(Molecule *mol, float *gpuout);

void cube_print(float *cube, Molecule *mol, const char *filename);
void cube_print_unwrap_ongpu(Molecule *mol, float *gpucube, const char *filename);
*/

extern "C" void qm_grid_toGPU(Grid *g);

#endif
