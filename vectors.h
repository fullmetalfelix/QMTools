
#ifndef VECTORS
#define VECTORS

/*typedef struct Vector3 Vector3;
struct Vector3 {
  double x,y,z;
};
*/

typedef union Vector3 Vector3;
union Vector3 {
	double data[3];
	struct {double x,y,z;};
};


Vector3 vectors_zero();
Vector3 vectors_copy(Vector3 v);

void vectors_add(Vector3 *v, Vector3 *u, Vector3 *dst);
void vectors_scale(Vector3 *v, double scale);
void vectors_opposite(Vector3 *v, Vector3 *dst);
void vectors_diff(Vector3 *v, Vector3 *u, Vector3 *dst);
double vectors_dot(Vector3 *u, Vector3 *v);
double vectors_norm2(Vector3 *v);
double vectors_norm(Vector3 *v);
double vectors_angle(Vector3 *u, Vector3 *v);
double vectors_cosangle(Vector3 *u, Vector3 *v);
void vectors_cross(Vector3 *v, Vector3 *u, Vector3 *result);

void vectors_mincomp(Vector3 *a, Vector3 *b, Vector3 *c);
void vectors_maxcomp(Vector3 *a, Vector3 *b, Vector3 *c);
double vectors_min(Vector3 *v);
double vectors_max(Vector3 *v);

void vectors_ceiled(Vector3 *a);


void vectors_normalise(Vector3 *v);
Vector3 vectors_normalised(Vector3 *v);
void vectors_normalise_with(Vector3 *v, double r);

void vectors_toRTP(Vector3 *v, Vector3 *dst);
void vectors_toRcosTP(Vector3 *v, Vector3 *dst);

void vectors_random_onsphere(Vector3 *dst);


void vectors_rotate_angleaxis(Vector3 *v, Vector3 *axis, double angle);

void vectors_print(Vector3 *v);

#endif
