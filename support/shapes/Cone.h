#ifndef CONE_H
#define CONE_H

#include "Shape.h"
#include "CircularPlane.h"

class Cone : public Shape
{
public:
    Cone(int slices, int columns);
    ~Cone();

private:
    int m_strips;
    int m_columns;
    std::unique_ptr<CircularPlane> circularPane;
};
#endif // CONE_H
