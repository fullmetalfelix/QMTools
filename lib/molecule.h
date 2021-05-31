#ifndef MOLECULE
#define MOLECULE



// maximum number of shells for one specie
#define MAXAOS 15
// maximum number of primitives contracted in a shell
#define MAXAOC 20
// extra space around atoms - in ANG
#define MOLFAT 2.0f


typedef struct BasisSet BasisSet;


typedef struct Molecule Molecule;
struct Molecule {

	int 	natoms;
	int		*types;
	float3	*coords; // in BOHR

	int		qtot;

	int 	norbs;
	float 	*dm, *d_dm;
	short4	*ALMOs, *d_ALMOs;

	BasisSet *basisset;

	int 	*d_types;
	float3 	*d_coords;

};


//extern "C" Molecule* molecule_new(int natm, int *Zs, float *coords);
//extern "C" Molecule* molecule_del(Molecule *mol);
extern "C" void molecule_init(Molecule *mol);
extern "C" void molecule_del(Molecule *mol);
extern "C" void molecule_densitygrid(Molecule *m, Grid *g, float step, float fat);




//void molecule_load_xyz(Molecule *mol, const char *filename);
//void molecule_clear(Molecule *mol);
//void molecule_free(Molecule *mol);

//void molecule_gpu_init(Molecule *mol);
//void molecule_gpu_free(Molecule *mol);

Molecule* molecule_load_complete(const char *path);
void molecule_free_complete(Molecule *m);

void molecule_write_bin(Molecule *mol, FILE *fbin);

//void molecule_basis_build(Molecule *mol, MolBasis *b);
//void molecule_basis_load_dm(MolBasis *bas, const char *filename);
//void molecule_basis_clear(MolBasis *b);
//void molecule_basis_free(MolBasis *b);


#endif
