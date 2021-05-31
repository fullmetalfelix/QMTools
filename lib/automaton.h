#ifndef AUTOMATON
#define AUTOMATON

#define Bp 9
#define BF (B+2)
#define BF_2 (BF * BF)
#define BF_3 (BF * BF * BF)
#define FBUF (BF_3*4)

#define BR 6
#define BR_2 (BR * BR)
#define BR_3 (BR * BR * BR)
#define RBUF (BR_3*4)


#define FlatBUF (BF*BF*3)
#define FlatBUF_TOT (FlatBUF*4)


#define OneOverSqrt2 0.7071067811865475
#define OneOverSqrt3 0.5773502691896258
#define OneOverDIFFTOT 0.05234482976098482

#define DIFFQ 0.2
#define DIFFA 0.2
#define DIFFB 0.2

#define DNASIZE 552


typedef struct Molecule Molecule;
typedef struct Grid Grid;


extern "C" void automaton_set_NN(float *parameters, int size);
extern "C" void automaton_reset(Grid *g);
extern "C" void automaton_compute_vnn(Grid *g, Molecule *m);
extern "C" void automaton_compute_qseed(Grid *g, Molecule *m);
//extern "C" void automaton_compute_evolution(Grid *g, Molecule *m);
extern "C" float automaton_compute_evolution(Grid *g, Molecule *m, int iterations, int copyback);
extern "C" float automaton_compare(Grid *g1, Grid *g2);


#endif
