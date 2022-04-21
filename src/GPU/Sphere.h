#ifndef SPHERE_H
#define SPHERE_H
#include "Vec.h"
/**
 * Description of a sphere using 3D-vector and radius
 * c - center
 * r - radius
 * col - color
 * ref - reflection
 */
class Sphere {
public:
    __host__ __device__ Sphere() : m_c(Vec()), m_r(0), m_col(Vec()), m_ref(0) {}
    __host__ __device__ Sphere(const Vec & c, const float r, const Vec & col, const float ref) : m_c(c), m_r(r), m_col(col), m_ref(ref) {}

    __host__ __device__ Vec c() const { return m_c; }
    __host__ __device__ float r() const { return m_r; }
    __host__ __device__ Vec col() const { return m_col; }
    __host__ __device__ float ref() const { return m_ref; }
private:
    Vec m_c;
    float m_r;
    Vec m_col;
    float m_ref;
};


#endif