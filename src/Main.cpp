#include <iostream>
#include <cmath>
#include "Vec.h"
#include "Ray.h"
#include "Sphere.h"

float intersect_sphere(const Sphere & s, const Ray & r) {
    Vec Ac = r.A() - s.c();
    float a = dot(r.B(), r.B());
    float b = 2 * dot(r.B(), Ac);
    float c = dot(Ac, Ac) - s.r() * s.r();
    float disc = b * b - 4 * a * c;
    if(disc > 0) {
        float dist_sqrt = sqrt(disc);
        float q = b < 0 ? (-b - dist_sqrt) / 2.0 : (-b + dist_sqrt) / 2.0;
        float t0 = q / a;
        float t1 = c / q;
        t0 = std::min(t0, t1);
        t1 = std::max(t0, t1);
        if(t1 >= 0) return t0 < 0 ? t1 : t0;
    }
    return -1;
}

Sphere SPHERE; 
Vec LIGHT;

Vec trace_ray(Ray & r) {
    float t = intersect_sphere(SPHERE, r);
    if(t == -1) return Vec();
    Vec M = r.P(t);
    Vec N = norm(M - SPHERE.c());
    Vec toL = norm(LIGHT - M);
    Vec toO = norm(r.A() - M);
    Vec col = Vec(0.05,0.05,0.05);
    col += 1 * std::max(dot(N, toL), 0.0f) * Vec(0,0,1);
    col += 1 * 50 * std::max(dot(N, norm(toL + toO)), 0.0f) * Vec(1,1,1);
    return col;
}

#define WIDTH 256
#define HEIGHT 256

int main() {
    SPHERE = Sphere(Vec(0,0,2), 1000);
    LIGHT = Vec(5,5,-10);
    Vec v1 = Vec();
    Vec v2 = Vec(0,0,2);
    Ray r = Ray(v1, v2);

    Vec test = trace_ray(r);
    //Ray r = Ray(Vec(), Vec(0,0,5));
    std::cout << test << std::endl;
    return 0;
}
