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

int main() {
    Sphere s = Sphere(Vec(0,0,2), 4);
    Ray r = Ray(Vec(), Vec(0,0,5));
    std::cout << intersect_sphere(s, r) << std::endl;
    return 0;
}
