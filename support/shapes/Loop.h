#ifndef LOOP_H
#define LOOP_H

#include "Surface.h"

class Loop
        :public Surface
{
public:
    Loop(float innerHeight, float radius, float angleOfTip, int columns, int strips);
    ~Loop();
    virtual void createTriangles() override;

    float innerHeight;
    float radius;
    float angleOfTip;
    int columns;
    int strips;
};


#endif // LOOP_H
