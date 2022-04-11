#ifndef RECTPLANE_H
#define RECTPLANE_H

#include "Surface.h"

//Creates a rectangle on the ZY plane with it's top left vertex on 0, 0
class RectPlane
        :public Surface
{
public:
    RectPlane(float width, float height, int cellsPerEdge);
    ~RectPlane();
    virtual void createTriangles() override;

    float width;
    float height;
    float cellsPerEdge;
};

#endif // RECTPLANE_H
