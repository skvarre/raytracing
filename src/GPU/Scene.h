#ifndef SCENE_H
#define SCENE_H

#include "Sphere.h"
#include "Vec.h"

Sphere * makeScene(int scase) {
    switch(scase) {
        Sphere * scene;
        case 1:
            cudaMallocManaged(&scene, sizeof(Sphere));
            scene[0] = Sphere(Vec( 0,  0, -1),  0.45, Vec(1.0, 0.000, 0.000), 0.5);
            return scene;
        case 2:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec( 0,  1, -1),  0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0, -1, -1),  0.45, Vec(0.0, 1.000, 0.000), 0.5);
            return scene;
        case 3:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec( 0,  0,  0),  0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0, -1, -1),  0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1, -1),  0.45, Vec(0.0, 0.000, 1.000), 0.5);
            return scene;
        case 4:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            return scene;
        case 5:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            scene[4] = Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5);
            return scene;
        case 6:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            scene[4] = Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5);
            scene[5] = Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5);
            return scene;
        case 7:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            scene[4] = Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5);
            scene[5] = Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5);
            scene[6] = Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5);
            return scene;
        case 8:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            scene[4] = Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5);
            scene[5] = Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5);
            scene[6] = Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5);
            scene[7] = Sphere(Vec(-1, -1,  -1), 0.45, Vec(0.6, 0.293, 0.000), 0.5);
            return scene;
        case 9:
            cudaMallocManaged(&scene, (int)scase*sizeof(Sphere));
            scene[0] = Sphere(Vec(-1,  0,  -1), 0.45, Vec(1.0, 0.000, 0.000), 0.5);
            scene[1] = Sphere(Vec( 0,  1,  -1), 0.45, Vec(0.0, 1.000, 0.000), 0.5);
            scene[2] = Sphere(Vec( 0, -1,  -1), 0.45, Vec(0.0, 0.000, 1.000), 0.5);
            scene[3] = Sphere(Vec( 1,  0,  -1), 0.45, Vec(1.0, 1.000, 0.000), 0.5);
            scene[4] = Sphere(Vec( 0,  0,  -1), 0.45, Vec(1.0, 0.640, 0.000), 0.5);
            scene[5] = Sphere(Vec( 1,  1,  -1), 0.45, Vec(0.9, 0.191, 0.387), 0.5);
            scene[6] = Sphere(Vec( 1, -1,  -1), 0.45, Vec(0.5, 0.000, 0.500), 0.5);
            scene[7] = Sphere(Vec(-1, -1,  -1), 0.45, Vec(0.6, 0.293, 0.000), 0.5);
            scene[8] = Sphere(Vec(-1,  1,  -1), 0.45, Vec(0.4, 0.410, 0.410), 0.5);
            return scene;
        default:
            return scene;
    }
}

#endif