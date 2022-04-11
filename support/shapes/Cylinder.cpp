#include "Cylinder.h"
#include <iostream>
#include <vector>
#include "GL/glew.h"

Cylinder::Cylinder(int param1, int param2) :
    m_strips(param1),
    m_columns(param2),
    loop(new Loop(1.f, 0.5f, M_PI_2, m_columns, m_strips)),
    plane(new CircularPlane(0.5f, 0.f, m_columns, m_strips))
{
    loop->createTriangles();
    loop->addPoints(m_vertexData);
    plane->createTriangles();
    plane->rotate(M_PI_2, glm::vec3(0, 1, 0));
    plane->translate(glm::vec3(0, 0.5f, 0));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI, glm::vec3(0, 0, 1));
    plane->rotate(M_PI, glm::vec3(0, 1, 0));
    plane->addPoints(m_vertexData);
    buildVAO();
}

Cylinder::~Cylinder()
{
}
