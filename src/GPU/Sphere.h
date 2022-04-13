#ifndef SPHERE_H
#define SPHERE_H
#include "Vec.h"
/**
 * Description of a sphere using 3D-vector and radius
 */
class Sphere {
public:
    __host__ __device__ Sphere() : m_c(Vec()), m_r(0) {}
    __host__ __device__ Sphere(const Vec & c, const float r) : m_c(c), m_r(r) {}

    __host__ __device__ Vec c() const { return m_c; }
    __host__ __device__ float r() const { return m_r; }
private:
    Vec m_c;
    float m_r;
};


#endif