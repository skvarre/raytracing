#ifndef SCENE_H
#define SCENE_H

#include "Sphere.h"
#include "Vec.h"

Sphere * makeScene(int scase) {
    switch(scase) {
        Sphere * scene;
        case 0:
            cudaMallocManaged(&scene, (int)4*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.7, Vec(0.0, 0.000, 1.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.7, Vec(0.5, 0.223, 0.500), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.7, Vec(1.0, 0.572, 0.184), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.7, Vec(0.0, 0.500, 1.000), 0.5);
            return scene;
        case 1:
            cudaMallocManaged(&scene, sizeof(Sphere));
            scene[0] = Sphere(Vec( 0,  0,  0),  1.0, Vec(1.0, 0.000, 0.000), 0.5);
            return scene;
        default:
            return scene;
    }
}

#endif