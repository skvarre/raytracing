#ifndef VEC_H
#define VEC_H
/**
 * Class describing a 3-dimensional vector 
 */
class Vec {
public:
    Vec() : m_x(0), m_y(0), m_z(0) {}
    Vec(float x, float y, float z) : m_x(x), m_y(y), m_z(z) {}

    float x() const { return m_x; }
    float y() const { return m_y; }
    float z() const { return m_z; }
private:
    float m_x;
    float m_y;
    float m_z;
}

#endif