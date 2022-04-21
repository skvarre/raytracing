#ifndef RAY_H
#define RAY_H
/**
 * Class describing a ray
 * Modified with __device__ keyword to be compiled on CUDA
 */
#include "Vec.h"

class Ray {
public:
    __host__ __device__ Ray() : m_A(Vec()), m_B(Vec()) {}
    __host__ __device__ Ray(const Vec & A, const Vec & B) : m_A(A), m_B(B) {}
    
    //P(t) = A + tB
    __host__ __device__ Vec P(float t) { return m_A + t*m_B; }

    __host__ __device__ Vec A() const { return m_A; }
    __host__ __device__ Vec B() const { return m_B; }

private:
    Vec m_A;
    Vec m_B;
};

#endif