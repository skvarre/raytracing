#ifndef SPHERE_H
#define SPHERE_H
#include "Vec.h"
/**
 * Description of a sphere using 3D-vector and radius
 */
class Sphere {
public:
    Sphere() : m_c(Vec()), m_r(0) {}
    Sphere(const Vec & c, const float r) : m_c(c), m_r(r) {}

    Vec c() const { return m_c; }
    float r() const { return m_r; }
private:
    Vec m_c;
    float m_r;
};


#endif