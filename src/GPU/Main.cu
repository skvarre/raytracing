#include "Vec.h"
#include "Ray.h"
#include "Sphere.h"
#include "Traced.h"
#include "Scene.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iterator>

#define WIDTH  800
#define HEIGHT 800

//Check intersection of ray and sphere, solve for t
__device__
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
    return INFINITY;
}

__device__
Traced trace_ray(Ray & r, Sphere * scene, Vec LIGHT) {
    float t = INFINITY;
    float t_object;
    int object_i = 0;
    
    for(int i = 0; i < 4; ++i) {
        t_object = intersect_sphere(scene[i], r);
        if(t_object < t) {
            t = t_object;
            object_i = i;
        }
    }
    if(t == INFINITY) {
        return Traced();
    }    
    Sphere object = scene[object_i];
    Vec M = r.P(t);
    Vec N = norm(M - object.c());
    Vec toL = norm(LIGHT - M);
    Vec toO = norm(r.A() - M);
    float l[3];
    int j = 0;
    for(int i = 0; i < 4; ++i) {
        if(i != object_i) {
            l[j] = intersect_sphere(object, Ray(M + 0.0001 * N, toL));
            ++j;
        }
    }    
    if(sizeof(l)/sizeof(*l) != 0 && *std::min_element(std::begin(l), std::end(l)) < INFINITY) {
        return Traced();
    }
    Vec col = Vec(0.05,0.05,0.05);
    // Shading osv.
    col += 1 * std::max(dot(N, toL), 0.0f) * object.col();
    col += pow((1 * std::max(dot(N, norm(toL + toO)), 0.0f) * Vec(1,1,1)), 50);
    return Traced(object, M, N, col);
}

__host__
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
void run(Vec * res, Sphere * scene, Vec LIGHT) {
    Vec O = Vec(0,0,2); // Camera position
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= WIDTH || j >= HEIGHT) return;
    int index = j*WIDTH + i;
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
        Traced traced = trace_ray(OD, scene, LIGHT);
        if(traced.m_col_ray.x() == -1 && traced.m_col_ray.y() == -1 && traced.m_col_ray.z() == -1) {
            break; 
        }
        Vec M = traced.m_M;
        Vec N = traced.m_N; 
        Vec col_ray = traced.m_col_ray;
        rayO = M + 0.0001 * N;
        rayD = norm(rayD - 2 * dot(rayD, N) * N);
        col += ref * col_ray;
        ref *= traced.m_sphere.ref();
        ++depth;
    }
    res[index] = col;
}

int main() {
    // Setup
    Sphere * scene = makeScene(0);
    Vec * res;
    Vec LIGHT = Vec(-5,-5,10);
    int blocks_x = 8;
    int blocks_y = 8;
    int N = HEIGHT * WIDTH;
    cudaMallocManaged(&res, N*sizeof(Vec));
    // Denna är lite spännande
    dim3 blocks(WIDTH/blocks_x+1, HEIGHT/blocks_y+1);
    dim3 threads(blocks_x, blocks_y);
    

    auto start = std::chrono::system_clock::now();
    
    run<<<blocks,threads>>>(res, scene, LIGHT);
    cudaDeviceSynchronize();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cerr << "GPU time: " << elapsed.count() << " seconds" << std::endl;
    

    // Pipe to file
    
    std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for(int i = 0; i < WIDTH; ++i) {
        for(int j = HEIGHT - 1; j >= 0; --j) {
            int index = j*WIDTH + i;
            std::cout << clip(res[index].x()) << ' ' << clip(res[index].y()) << ' ' << clip(res[index].z()) << '\n';
        }
    }
    
    // Cleanup
    cudaFree(scene);
    cudaFree(res);

    return 0;
}
