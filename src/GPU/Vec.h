#ifndef VEC_H
#define VEC_H

#include <iostream>
#include <cmath>

/**
 * Class describing a 3-dimensional vector 
 */
class Vec {
public:
    __host__ __device__ Vec() : m_x(0), m_y(0), m_z(0) {}
    __host__ __device__ Vec(float x, float y, float z) : m_x(x), m_y(y), m_z(z) {}

    //Accessors
    __host__ __device__ float x() const { return m_x; }
    __host__ __device__ float y() const { return m_y; }
    __host__ __device__ float z() const { return m_z; }

    __host__ __device__ void x(float x) { m_x = x; }
    __host__ __device__ void y(float y) { m_y = y; }
    __host__ __device__ void z(float z) { m_z = z; }
     
    __host__ __device__ Vec operator-() const { return Vec(-m_x, -m_y, -m_z); }
    __host__ __device__ Vec operator+=(const Vec & v) {
        m_x += v.m_x; 
        m_y += v.m_y;
        m_z += v.m_z;
        return *this;
    }

    //Kanske inte behövs här, utan snarare längre ner? Den som lever får se. 
    __host__ __device__ float length();

private:
    float m_x;
    float m_y;
    float m_z;
};


std::ostream& operator<<(std::ostream &os, const Vec &v){
    return os << "(" << v.x() << "," << v.y() << "," << v.z() << ")";
}

__host__ __device__
Vec operator-(const Vec &lhs, const Vec &rhs){
    return Vec(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
}

__host__ __device__
Vec operator+(const Vec &lhs, const Vec &rhs) {
    return Vec(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
}

__host__ __device__
Vec operator*(float t, const Vec &v) {
    return Vec(t*v.x(), t*v.y(), t*v.z());
}

__host__ __device__
Vec pow(const Vec &v, float x) {
    return Vec(pow(v.x(),x), pow(v.y(),x), pow(v.z(),x));
}

__host__ __device__
float dot(const Vec & lhs, const Vec & rhs) {
    return lhs.x()*rhs.x() + lhs.y()*rhs.y() + lhs.z()*rhs.z();
}

__host__ __device__
float Vec::length() {
    return sqrt(m_x*m_x + m_y*m_y + m_z*m_z);
}

__host__ __device__
Vec norm(Vec v) {
    return 1/v.length() * v;
}


#endif