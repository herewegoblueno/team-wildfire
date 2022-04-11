#ifndef CUBE_H
#define CUBE_H

#include "Shape.h"
#include "Rectplane.h"

class Cube : public Shape
{
public:
    Cube(int param1);
    ~Cube();

private:
    int cellsPerSide;
    std::unique_ptr<RectPlane> plane;
};

#endif // CUBE_H
