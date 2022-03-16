#ifndef VEC_H
#define VEC_H

/**
 * Class describing a 3-dimensional vector 
 */
class Vec {
public:
    Vec() : m_x(0), m_y(0), m_z(0) {}
    Vec(float x, float y, float z) : m_x(x), m_y(y), m_z(z) {}

    //Accessors
    float x() const { return m_x; }
    float y() const { return m_y; }
    float z() const { return m_z; }

    void x(float x) { m_x = x; }
    void y(float y) { m_y = y; }
    void z(float z) { m_z = z; }


private:
    float m_x;
    float m_y;
    float m_z;
};

Vec operator-(const Vec &lhs, const Vec &rhs){
    return Vec(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
}

Vec operator+(const Vec &lhs, const Vec &rhs) {
    return Vec(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
}

Vec operator*(float t, const Vec &v) {
    return Vec(t*v.x(), t*v.y(), t*v.z());
}

#endif