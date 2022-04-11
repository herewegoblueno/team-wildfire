#include "Cone.h"
#include <iostream>
#include <vector>
#include "GL/glew.h"

Cone::Cone(int param1, int param2) :
    m_strips(param1),
    m_columns(param2),
    circularPane(new CircularPlane(0.5f, 1, m_columns, m_strips))
{

    circularPane->createTriangles();
    circularPane->translate(glm::vec3(0, -0.5f, 0));
    circularPane->addPoints(m_vertexData);
    circularPane->cleanup();
    circularPane->height = 0;
    circularPane->createTriangles();
    circularPane->rotate(M_PI, glm::vec3(0, 0, 1));
    circularPane->translate(glm::vec3(0, -0.5f, 0));
    circularPane->addPoints(m_vertexData);
    buildVAO();
}

Cone::~Cone()
{
}
