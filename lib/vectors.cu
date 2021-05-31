/*! @file vectors.c
  @brief Vector3 calculation tools.

*/

#include "vectors.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>




Vector3 vectors_zero() {
	Vector3 result;
	result.x = result.y = result.z = 0;
	return result;
}
/*! \brief Compute the opposite of a vector v and store it in dst.
 * \param v is a reference to the original vector.
 * \param dst is where the result will go.
 * ****************************************************************** */
void vectors_opposite(Vector3 *v, Vector3 *dst) {
	dst->x = -v->x;
	dst->y = -v->y;
	dst->z = -v->z;
}


void vectors_scale(Vector3 *v, double scale) {
	v->x *= scale;
	v->y *= scale;
	v->z *= scale;
}

void vectors_add(Vector3 *v, Vector3 *u, Vector3 *dst) {
	dst->x = u->x + v->x;
	dst->y = u->y + v->y;
	dst->z = u->z + v->z;
}


void vectors_cross(Vector3 *v, Vector3 *u, Vector3 *dst) {
	dst->x = v->y*u->z - v->z*u->y;
	dst->y = v->z*u->x - v->x*u->z;
	dst->z = v->x*u->y - v->y*u->x;
}

/*! \brief Compute the magnitude of a vector v.
 * \param v is a reference to the vector.
 * \return the norm of the vector.
 * ****************************************************************** */
double vectors_norm(Vector3 *v) {

  return sqrt(vectors_norm2(v));
}

void vectors_normalise(Vector3 *v) {
	double r = vectors_norm(v);
	if(r > 0) {
		v->x /= r;
		v->y /= r;
		v->z /= r;
	} else {
		printf("VECTORS ERROR: norm is 0, cannot normalise!\n");
	}
}

void vectors_normalise_with(Vector3 *v, double r) {
	if(r > 0) {
		v->x /= r;
		v->y /= r;
		v->z /= r;
	} else {
		printf("VECTORS ERROR: norm is 0, cannot normalise!\n");
	}
}

Vector3 vectors_normalised(Vector3 *v) {
	
	Vector3 result;
	result.x = v->x;
	result.y = v->y;
	result.z = v->z;
	
	double r = vectors_norm(v);
	
	if(r > 0) {
		result.x /= r;
		result.y /= r;
		result.z /= r;
	} else {
		printf("VECTORS ERROR: norm is 0, cannot be normalised!\n");
	}
	return result;
}

/*! \brief Compute the squared magnitude of a vector v.
 * \param v is a reference to the original vector.
 * \return the squared norm of the vector.
 * ****************************************************************** */
double vectors_norm2(Vector3 *v) {
	double result = v->x * v->x;
	result += v->y * v->y;
	result += v->z * v->z;
	return result;
}

/*! \brief Compute the scalar product of two vectors.
 * \param v is a reference to a vector.
 * \param u is a reference to the other vector.
 * \return v.u
 * ****************************************************************** */
double vectors_dot(Vector3 *u, Vector3 *v) {
	double dot = u->x * v->x;
	dot += u->y * v->y;
	dot += u->z * v->z;
	return dot;
}



double vectors_angle(Vector3 *u, Vector3 *v) {
	
	double dot = vectors_dot(u, v);
	dot /= vectors_norm(u) * vectors_norm(v);
	return acos(dot);
}

double vectors_cosangle(Vector3 *u, Vector3 *v) {
	
	double dot = vectors_dot(u, v);
	dot /= vectors_norm(u) * vectors_norm(v);
	return dot;
}

/*! \brief Converts v to spherical coordinates.
 * The resulting xyz component of the vector will contain R, theta and phi.
 * \param v is a reference to a vector to convert.
 * \param dst pointer for the result.
 * ****************************************************************** */
void vectors_toRTP(Vector3 *v, Vector3 *dst) {
	
	dst->x = vectors_norm(v);
	dst->y = (dst->x == 0)? 0 : acos(v->z/dst->x);
	dst->z = (dst->x == 0)? 0 : atan2(v->y, v->x);
}

/*! \brief Converts v to spherical coordinates.
 * The resulting xyz component of the vector will contain R, cos(theta) and phi.
 * \param v is a reference to a vector to convert.
 * \param dst pointer for the result.
 * ****************************************************************** */
void vectors_toRcosTP(Vector3 *v, Vector3 *dst) {
	dst->x = vectors_norm(v);
	dst->y = (dst->x == 0)? 0 : v->z/dst->x;
	dst->z = (dst->x == 0)? 0 : atan2(v->y, v->x);
}

/*! \brief Compute the difference between two vectors.
 * \param v is a reference to a vector.
 * \param u is a reference to the other vector.
 * \param dst pointer for the result: v-u.
 * ****************************************************************** */
void vectors_diff(Vector3 *v, Vector3 *u, Vector3 *dst) {
  dst->x = v->x - u->x;
  dst->y = v->y - u->y;
  dst->z = v->z - u->z;
}



void vectors_random_onsphere(Vector3 *dst) {
	
	double theta = 3.1415926535897932384626433832795028841971693993751058209749445923078164062*(double)(rand())/RAND_MAX;
	double phi = 2*3.1415926535897932384626433832795028841971693993751058209749445923078164062*(double)(rand())/RAND_MAX;
	
	dst->x = sin(theta)*cos(phi);
	dst->y = sin(theta)*sin(phi);
	dst->z = cos(theta);
	
}



void vectors_rotate_angleaxis(Vector3 *v, Vector3 *axis, double angle) {
	
	vectors_normalise(axis);
	
	Vector3 tmp = *v; 
	Vector3 cross; vectors_cross(axis, v, &cross);
	vectors_scale(&cross, sin(angle));
	vectors_scale(&tmp, cos(angle));
	Vector3 tmp2;
	vectors_add(&tmp, &cross, &tmp2);
	tmp = *axis;
	vectors_scale(&tmp, vectors_dot(axis, v) * (1-cos(angle)));
	vectors_add(&tmp2, &tmp, v);
}

void vectors_ceiled(Vector3 *a) {

	a->x = ceil(a->x);
	a->y = ceil(a->y);
	a->z = ceil(a->z);
}


void vectors_print(Vector3 *v) {
	printf("[%lf, %lf, %lf]\n",v->x, v->y, v->z);
}




float3 float3_max(float3 a, float3 b) {

	float3 result = a;
	if(b.x > a.x) result.x = b.x;
	if(b.y > a.y) result.y = b.y;
	if(b.z > a.z) result.z = b.z;

	return result;
}

float3 float3_min(float3 a, float3 b) {

	float3 result = a;
	if(b.x < a.x) result.x = b.x;
	if(b.y < a.y) result.y = b.y;
	if(b.z < a.z) result.z = b.z;

	return result;
}

float3 float3_add(float3 v, float c) {
	float3 result = v;
	v.x += c;
	v.y += c;
	v.z += c;
	return result;
}

float3 float3_div(float3 v, float c) {
	float3 result = v;
	v.x /= c;
	v.y /= c;
	v.z /= c;
	return result;
}


float3 float3_ceiled(float3 v) {

	float3 a;
	a.x = ceil(v.x);
	a.y = ceil(v.y);
	a.z = ceil(v.z);
	return a;
}


__host__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__host__ float3 operator+(const float3 &a, const float &b) {

	return make_float3(a.x+b, a.y+b, a.z+b);
}

__host__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__host__ float3 operator-(const float3 &a, const float &b) {

	return make_float3(a.x-b, a.y-b, a.z-b);
}

__host__ float3 operator*(const float3 &a, const float &b) {

	return make_float3(a.x*b, a.y*b, a.z*b);
}
__host__ float3 operator/(const float3 &a, const float &b) {

	return make_float3(a.x/b, a.y/b, a.z/b);
}

