#ifndef SPHERE_H
#define SPHERE_H

#include "Shape.h"
#include "HemispherePlane.h"

class Sphere : public Shape
{
public:
    Sphere(int param1, int param2);
    ~Sphere();

private:
    int m_strips;
    int m_columns;
    std::unique_ptr<HemispherePlane> hemisphere;
};

#endif // SPHERE_H
