#ifndef TRACED_H
#define TRACED_H
#include "Vec.h"
#include "Sphere.h"
/**
 * This ain't python
 */
class Traced {
public:
    Traced() : m_sphere(Sphere()), m_M(Vec()), m_N(Vec()), m_col_ray(Vec(-1,-1,-1)) {}
    Traced(Sphere sphere, Vec M, Vec N, Vec col_ray) : m_sphere(sphere), m_M(M), m_N(N), m_col_ray(col_ray) {}
    Sphere m_sphere;
    Vec m_M;
    Vec m_N;
    Vec m_col_ray;
};

#endif