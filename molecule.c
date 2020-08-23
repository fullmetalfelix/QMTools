#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "convolve.h"




// UNUSED
void Molecule_load_bin(Molecule *mol, const char *filename) {

	printf("opening molecule: %s\n", filename);

	int ri;
	double energy;

	// FILES FROM OLD RUNS ARE IN BOHR!
	// NEW RUNS FROM THE CCSD-CID database are in ANGSTROMS!

	FILE *fbin = fopen(filename, "rb");
	fread(&ri, sizeof(int), 1, fbin); mol->natoms = ri;
	fread(&energy, sizeof(double), 1, fbin);

	mol->qtot = 0;
	mol->atoms = malloc(sizeof(Atom) * mol->natoms);
	for(int i=0; i<mol->natoms; i++) {
		fread(&mol->atoms[i].Z, sizeof(int), 1, fbin);
		fread(&mol->atoms[i].position.x, sizeof(double), 1, fbin);
		fread(&mol->atoms[i].position.y, sizeof(double), 1, fbin);
		fread(&mol->atoms[i].position.z, sizeof(double), 1, fbin);
		//molecule[i].basis = &(basisset[basisOffset[molecule[i].Z]]);

		mol->qtot += mol->atoms[i].Z;
		
		printf("read atom %i\t%i\t%+5.3lf %+5.3lf %+5.3lf\n", i, mol->atoms[i].Z, 
			mol->atoms[i].position.x, mol->atoms[i].position.y, mol->atoms[i].position.z);

	}

	fclose(fbin);
}


/// this is for new CCSD-CID molecules
void Molecule_load_xyz(Molecule *mol, const char *filename) {

	printf("opening molecule: %s\n", filename);

	int natm = 0;
	int ri;
	double energy;
	char buffer[256];

	// FILES FROM OLD RUNS ARE IN BOHR!
	// NEW RUNS FROM THE CCSD-CID database are in ANGSTROMS!

	FILE *fbin = fopen(filename, "r");
	while(!feof(fbin)) {
		char ch = fgetc(fbin);
		if(ch == '\n')
			natm++;
	}
	fseek(fbin, 0, SEEK_SET); // rewind

	mol->natoms = natm;
	mol->posMin = vectors_zero();
	mol->posMax = vectors_zero();

	mol->qtot = 0;
	mol->atoms = malloc(sizeof(Atom) * mol->natoms);
	for(int i=0; i<mol->natoms; i++) {
		
		fscanf(fbin, "%i %lf %lf %lf\n", &mol->atoms[i].Z, &mol->atoms[i].position.x, &mol->atoms[i].position.y, &mol->atoms[i].position.z);

		vectors_scale(&mol->atoms[i].position, ANG2BOR); // convert back to bohr

		mol->qtot += mol->atoms[i].Z;
		
		vectors_mincomp(&mol->posMin, &mol->atoms[i].position, &mol->posMin);
		vectors_maxcomp(&mol->posMax, &mol->atoms[i].position, &mol->posMax);

		printf("read atom %i\t%i\t%+5.3lf %+5.3lf %+5.3lf\n", i, mol->atoms[i].Z, 
			mol->atoms[i].position.x, mol->atoms[i].position.y, mol->atoms[i].position.z);
	}

	fclose(fbin);
}



void Molecule_basis_build(Molecule *mol) {


	int nbas = 0;

	// dry run first
	for(int a=0; a<mol->natoms; a++) {
		
		Atom *atom = mol->atoms + a;
		AOBasis *atomBasis = global_basisset + global_atombasis_offset[atom->Z];

		for(int b=0; b<atomBasis->norbs; b++) {
			nbas += 2*atomBasis->orbs[b].L + 1;
		}
	}

	mol->norbs = nbas;
	printf("# of basis orbitals: %i\n",nbas);
	mol->orbitals = malloc(sizeof(Orbital) * nbas);
	Orbital *orb = mol->orbitals;

	for(int a=0; a<mol->natoms; a++) {
		
		Atom *atom = mol->atoms + a;
		AOBasis *atomBasis = global_basisset + global_atombasis_offset[atom->Z];

		for(int b=0; b<atomBasis->norbs; b++) {


			// loop over m values
			for(int mi=0; mi<(2*atomBasis->orbs[b].L + 1); mi++) {
				
				orb->atomIndex = a;
				orb->atom = atom;
				orb->ao = atomBasis->orbs + b;
				orb->m = global_m_values[orb->ao->L][mi];
				//printf("%i %i %i -- %i\n",a,b,mi, orb->m);
				orb++;
			}
		}
	}
	printf("basis orbitals allocated.\n");
	mol->actives = malloc(sizeof(unsigned short*) * nbas);
	mol->nactives = 0;
}

void Molecule_clear(Molecule *mol) {

	if(mol->atoms) free(mol->atoms);
	if(mol->orbitals) free(mol->orbitals);
	if(mol->actives) free(mol->actives);
}


void Molecule_box_get(Molecule *mol, Grid *grid) {

	grid->voxvol = grid->step * grid->step * grid->step;

	Vector3 *min = &grid->min;
	Vector3 *max = &grid->max;
	double fat = 0;

	for(int i=0; i<3; i++) {
		min->data[i] = DBL_MAX;
		max->data[i] = -DBL_MAX;
	}

	// determine the min/max for atom coords
	for (int a = 0; a < mol->natoms; ++a) {
		for(int i=0; i<3; i++) {
			double x = mol->atoms[a].position.data[i];
			min->data[i] = (x < min->data[i])? x : min->data[i];
			max->data[i] = (x > max->data[i])? x : max->data[i];
		}
	}

	for(int i=0; i<mol->norbs; i++) {

		fat = (mol->orbitals[i].ao->maxR2 > fat)? mol->orbitals[i].ao->maxR2 : fat;
	}
	fat = sqrt(fat);

	for(int i=0; i<3; i++) {
		min->data[i] -= fat;
		max->data[i] += fat;
	}

	// compute subdivisions
	vectors_diff(max, min, &grid->size);

	printf("grid fat: %lf \n", fat);
	printf("grid min: "); vectors_print(min);
	printf("grid max: "); vectors_print(max);
	printf("grid size: "); vectors_print(&grid->size);

	unsigned int np = 1;
	for(int i=0; i<3; i++) {
		grid->subdivs[i] = (unsigned int)ceil(grid->size.data[i] / grid->step);
		max->data[i] = min->data[i] + grid->subdivs[i] * grid->step;
		np *= grid->subdivs[i];
	}
	grid->npts = np;
	printf("grid subdivisions: %i %i %i - total: %i\n", grid->subdivs[0], grid->subdivs[1], grid->subdivs[2], np);

}
