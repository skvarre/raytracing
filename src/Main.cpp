#include <iostream>
#include "Vec.h"
#include "Ray.h"

int main() {
    std::cout << Vec(1,2,3) - Vec(1,2,4) << std::endl;
    return 0;
}
