#ifndef CYLINDER_H
#define CYLINDER_H

#include "Shape.h"
#include "Loop.h"
#include "CircularPlane.h"

class Cylinder : public Shape
{
public:
    Cylinder(int param1, int param2);
    ~Cylinder();

private:
    int m_strips;
    int m_columns;
    std::unique_ptr<Loop> loop;
    std::unique_ptr<CircularPlane> plane;
};

#endif // CYLINDER_H
