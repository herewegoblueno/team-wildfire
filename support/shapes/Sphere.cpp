#include "Sphere.h"
#include <iostream>
#include <vector>
#include "GL/glew.h"

Sphere::Sphere(int param1, int param2) :
    m_strips(param1),
    m_columns(param2),
    hemisphere (new HemispherePlane(0.5, m_columns, m_strips))
{

    hemisphere->createTriangles();
    hemisphere->addPoints(m_vertexData);
    hemisphere->rotate(M_PI, glm::vec3(1, 0, 0));
    hemisphere->rotate(M_PI / (float)hemisphere->columns * (hemisphere->columns % 2), glm::vec3(0, 1, 0));
    hemisphere->addPoints(m_vertexData);
    buildVAO();
}

Sphere::~Sphere()
{
}
