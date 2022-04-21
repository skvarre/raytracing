#include <iostream>
#include <cmath>
#include "Vec.h"
#include "Ray.h"
#include "Sphere.h"
#include "Traced.h"
#include <time.h>
#include <vector>
#include <algorithm>

#define WIDTH 400
#define HEIGHT 400

Sphere SPHERE; 
Vec LIGHT;
std::vector<Sphere> scene;

//Check intersection of ray and sphere, solve for t
float intersect_sphere(const Sphere & s, const Ray & r) {
    Vec Ac = r.A() - s.c(); // A = O, c = S
    float a = dot(r.B(), r.B()); // B = D
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
    return INFINITY;
}

Traced trace_ray(Ray & r) {
    float t = INFINITY;
    float t_object;
    int object_i = 0;
    int i = 0;
    for(Sphere object : scene) {
        t_object = intersect_sphere(object, r);
        if(t_object < t) {
            t = t_object;
            object_i = i;
        }
        ++i;
    }
    if(t == INFINITY) {
        return Traced();
    }
    Sphere object = scene[object_i];
    Vec M = r.P(t);
    Vec N = norm(M - object.c());
    Vec toL = norm(LIGHT - M);
    Vec toO = norm(r.A() - M);
    std::vector<float> l;
    int j = 0;
    for(Sphere object : scene) {
        if(j != object_i) {
            l.push_back(intersect_sphere(object, Ray(M + 0.0001 * N, toL)));
        }
        ++j;
    }
    if(l.size() != 0 && *std::min_element(l.begin(), l.end()) < INFINITY) {
        return Traced();
    }
    Vec col = Vec(0.05,0.05,0.05);
    //Shading osv.
    col += 1 * std::max(dot(N, toL), 0.0f) * object.col(); //Detta betyder färg 
    col += pow((1 * std::max(dot(N, norm(toL + toO)), 0.0f) * Vec(1,1,1)), 50);
    return Traced(object, M, N, col);

    /* WORKING CODE
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
    */
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

void run() {
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    Vec O = Vec(0,0,2);
    for(int i = 0; i < WIDTH; ++i) {
        for(int j = HEIGHT - 1; j >= 0; --j) {
            Vec col;
            float I = -1.0 + (2.0*i/(WIDTH-1.0));
            float J = -1.0 + (2.0*j/(HEIGHT-1.0));
            Vec D = norm(Vec(I,J,0) - O);
            Vec rayO = O;
            Vec rayD = D;
            int depth = 0;
            float ref = 1;
            while(depth < 5) {
                Ray OD = Ray(rayO, rayD);
                Traced traced = trace_ray(OD);
                if(traced.m_col_ray.x() == -1 && traced.m_col_ray.y() == -1 && traced.m_col_ray.z() == -1) {
                    break; 
                }
                Sphere object = traced.m_sphere;
                Vec M = traced.m_M;
                Vec N = traced.m_N; 
                Vec col_ray = traced.m_col_ray;
                rayO = M + 0.0001 * N;
                rayD = norm(rayD - 2 * dot(rayD, N) * N);
                col += ref * col_ray;
                ref *= traced.m_sphere.ref();
                ++depth;
            }
            std::cout << clip(col.x()) << ' ' << clip(col.y()) << ' ' << clip(col.z()) << '\n';
        } 
    }
}

int main() {
    //SPHERE = Sphere(Vec(0,0,-1), 1);
    LIGHT = Vec(-5,-5,10);/*
    for(int i = 0; i < 1000; i++){
        scene.push_back(Sphere(Vec(i*0.1,2*0.5*i*0.5,-3), .1, Vec(i/256,4,1), 0.5));  
    }*/
    scene.push_back(Sphere(Vec(-1,  0,  -1), .7, Vec(0.0, 0.000, 1.000), 0.5));  
    scene.push_back(Sphere(Vec( 0,  1,  -1), .7, Vec(0.5, 0.223, 0.500), 0.5));
    scene.push_back(Sphere(Vec( 0, -1,  -1), .7, Vec(1.0, 0.572, 0.184), 0.5));
    scene.push_back(Sphere(Vec( 1,  0,  -1), .7, Vec(0.0, 0.500, 1.000), 0.5));

    clock_t start, stop;
    start = clock();
    run();
    stop = clock();
    double time_seconds = ((double)(stop - start) / CLOCKS_PER_SEC);
    std::cerr << "CPU time: " << time_seconds << " SEGUNDOS SEÑOR" << std::endl;
    
    return 0;
}
