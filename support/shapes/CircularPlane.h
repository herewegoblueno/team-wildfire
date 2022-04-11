#ifndef CIRCULARPLANE_H
#define CIRCULARPLANE_H

#include "Surface.h"

class CircularPlane
        :public Surface
{
public:
    CircularPlane(float radius, float height, int sectors, int strips);
    ~CircularPlane();
    virtual void createTriangles() override;

    float radius;
    float height;
    int sectors;
    int strips;
};

#endif // CIRCULARPLANE_H
