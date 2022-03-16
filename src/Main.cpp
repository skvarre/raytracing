#include <iostream>
#include "Vec.h"
#include "Ray.h"

int main() {
    Vec v1(2,3,4);
    Vec v2(1,1,1);
    Ray r(v1, v2);
    std::cout << r.P(3) << std::endl;

    return 0;
}
