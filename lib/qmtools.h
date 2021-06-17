#ifndef SCSFDEF
#define SCSFDEF

#include <curand.h>
#include <curand_kernel.h>


#define ANG2BOR 1.8897259886

// maximum number of shells for one specie
#define MAXAOS 15
// maximum number of primitives contracted in a shell
#define MAXAOC 20
// extra space around atoms - in ANG
#define MOLFAT 2.0f

#define B 8
#define B_2 (B*B)
#define B_3 (B*B*B)

#define LB 512

#define DMTILE  16
#define DMTILE_2 256
#define twoDMTILE 32

#define NTYPES 10


typedef struct BasisSet BasisSet;
typedef struct QMTools QMTools;
typedef struct Molecule Molecule;


//extern __constant__ int c_esph_ns[2];
extern __constant__ int c_types[NTYPES];
extern int nTypes;


struct QMTools {

	int 	nrpts;
	int 	nacsf;

	float 	*qube;
	float 	*acsf;
	float3 	*rpts;

	curandState_t *d_state;
};


typedef struct SCSFGPU SCSFGPU;
struct SCSFGPU {

	int			*Zs;
	float3 		*coords;
	float 		*dm;
	short4 		*almos;

	float 		*qube;
	float 		*acsf;
	float3 		*rpts;

	float 		*VNe000;
	float 		*VNe00z;
	float 		*VNe0yz;
	float 		*VNexyz;
};


typedef struct Grid Grid;
struct Grid {

	dim3 		shape;
	dim3 		GPUblocks;
	uint 		npts;
	uint 		nfields;

	float3 		origin;
	float3 		Ax, Ay, Az;
	float 		step;

	float 		*qube;
	float 		*d_qube;
};


extern "C" void qm_ini(QMTools *obj);
extern "C" void qm_del(QMTools *obj);

extern "C" void qm_grid_ini(Grid *obj);
extern "C" void qm_grid_del(Grid *obj);

extern "C" void qm_densityqube(Molecule *m, Grid *g);
extern "C" void qm_densityqube_shmem(Molecule *m, Grid *g);
extern "C" void qm_densityqube_subgrid(Molecule *m, Grid *g);

extern "C" void qm_hartree(Molecule *m, Grid *q, Grid *v);



extern "C" void qm_gridmol_write(Grid *g, Molecule *m, const char* filename);

//extern "C" void scsf_compute(QMToold *obj, Molecule *mol);




#endif
