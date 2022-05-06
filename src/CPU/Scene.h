#ifndef SCENE_H
#define SCENE_H

#include "Sphere.h"
#include "Vec.h"
#include <vector>

void makeScene(int scase, std::vector<Sphere> & scene) {
    switch(scase) {
        case 1:
            scene.push_back(Sphere(Vec( 0,  0, -1),  0.45, Vec(1.0, 0.000, 0.000), 0.5));
            break;
        case 2:
            scene.push_back(Sphere(Vec( 0,  1, -1),  0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1, -1),  0.45, Vec(0.0, 1.000, 0.000), 0.5));
            break;
        case 3:
            scene.push_back(Sphere(Vec( 0,  1, -1),  0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1, -1),  0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0, -1),  0.45, Vec(0.0, 0.000, 1.000), 0.5));
            break;
        case 4:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            break;
        case 5:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5));
            break;
            
        case 6:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5));
            break;
        case 7:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5));
            scene.push_back(Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5));
            break;
        case 8:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5));
            scene.push_back(Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5));
            scene.push_back(Sphere(Vec(-1, -1,  -1), 0.45, Vec(0.6, 0.293, 0.000), 0.5));
            break;
        case 9:
            scene.push_back(Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5));
            scene.push_back(Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5));
            scene.push_back(Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5));
            scene.push_back(Sphere(Vec(-1, -1,  -1), 0.45, Vec(0.6, 0.293, 0.000), 0.5));
            scene.push_back(Sphere(Vec(-1,  1,  -1), 0.45, Vec(0.4, 0.410, 0.410), 0.5));
            break;
        default:
            break;
    }
}

#endif