#ifndef BASISSET
#define BASISSET



typedef struct BasisSet BasisSet;
struct BasisSet {

	// one for each species
	int *atomOffset;	// index of the atom basis in the shellOffset/Ls
	int *nshells;		// number of shells for atom[Z]

	// one for each shell
	int *Ls;
	int *shellOffset; 	// index of the shell parameters in alphas/coeffs

	// one for each primitive
	int nparams;
	float *alphas;
	float *coeffs;

	int **global_m_values;

	float 	*d_alphas;
	float 	*d_coeffs;
};


extern "C" void basisset_ini(BasisSet *basisset, const char *filename);
extern "C" void basisset_del(BasisSet *basisset);


//void scsf_basisset_load(SCSF *obj, const char *filename);
//void scsf_basisset_free(SCSF *obj);


#endif
