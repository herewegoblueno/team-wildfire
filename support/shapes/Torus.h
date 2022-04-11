#ifndef TORUS_H
#define TORUS_H

#include "Shape.h"
#include "Loop.h"

class Torus : public Shape
{
public:
    Torus(int param1, int param2, int param3);
    ~Torus();

private:
    int sides;
    int sideSmoothness;
    int thickness;
    float radius;
    std::unique_ptr<Loop> loop;
};

#endif // TORUS_H
