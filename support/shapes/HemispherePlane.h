#ifndef HEMISPHEREPLANE_H
#define HEMISPHEREPLANE_H

#include "Surface.h"

class HemispherePlane
        :public Surface
{
public:
    HemispherePlane(float radius, int columns, int rows);
    ~HemispherePlane();
    virtual void createTriangles() override;

    float radius;
    int columns;
    int rows;
};

#endif // HEMISPHEREPLANE_H
