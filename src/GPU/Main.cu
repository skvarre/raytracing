#include <iostream>
#include <cmath>
#include "Vec.h"
#include "Ray.h"
#include "Sphere.h"

#define WIDTH 400
#define HEIGHT 400

//Check intersection of ray and sphere, solve for t
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
        float temp = t0;
        t0 = std::min(t0, t1);
        t1 = std::max(temp, t1);
        if(t1 >= 0) return t0 < 0 ? t1 : t0;
    }
    return -1;
}

__constant__ Sphere SPHERE;
__constant__ Vec LIGHT;

__device__
Vec trace_ray(Ray & r) {
    float t = intersect_sphere(SPHERE, r);
    if(t == -1) return Vec(-1,-1,-1);
    Vec M = r.P(t);
    Vec N = norm(M - SPHERE.c());
    Vec toL = norm(LIGHT - M);
    Vec toO = norm(r.A() - M);
    Vec col = Vec(0.05,0.05,0.05);
    col += 1 * std::max(dot(N, toL), 0.0f) * Vec(0,0,1);
    col += pow((1 * std::max(dot(N, norm(toL + toO)), 0.0f) * Vec(1,1,1)), 50);
    return col;
}

float clip(float f) {
    if(f < 0.0) {
        return 0.0;
    }
    if(f > 1.0) {
        return 1.0*255.999;
    }
    return f*255.999;
}

__global__
void run() {
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    Vec O = Vec(0,0,1);
    for(int i = 0; i < WIDTH; ++i) {
        for(int j = 0; j < HEIGHT; ++j) {
            float I = -1.0 + (2.0*i/(WIDTH-1.0));
            float J = -1.0 + (2.0*j/(HEIGHT-1.0));
            Ray r = Ray(O, norm(Vec(I,J,0) - O));
            Vec col = trace_ray(r);
            std::cout << clip(col.x()) << ' ' << clip(col.y()) << ' ' << clip(col.z()) << '\n';
        }
    }
}

int main() {
    SPHERE = Sphere(Vec(0,0,-1), 1);
    LIGHT = Vec(-5,-5,10);
    Ray r = Ray(Vec(0,0,1), Vec(0,0,5));
    float test = intersect_sphere(SPHERE, r);
    //std::cout << pow(Vec(2,3,1), 2) << std::endl;

    run<<<1,1>>>();
    
    return 0;
}