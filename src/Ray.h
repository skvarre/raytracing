#ifndef RAY_H
#define RAY_H
/**
 * Class describing a ray
 */
#include "Vec.h"

class Ray {
public:
    Ray() : m_A(Vec()), m_B(Vec()) {}
    Ray(const Vec & A, const Vec & B) : m_A(A), m_B(B) {}
    
    //P(t) = A + tB
    Vec P(float t) { return m_A + t*m_B; }

private:
    Vec m_A;
    Vec m_B;
};

#endif