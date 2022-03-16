#include <iostream>
#include "Vec.h"
#include "Ray.h"

int main() {
    std::cout << dot(Vec(1,2,2), Vec(2,3,3)) << std::endl;
    return 0;
}
