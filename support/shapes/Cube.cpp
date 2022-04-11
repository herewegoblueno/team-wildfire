#include "Cube.h"
#include <iostream>
#include <vector>
#include "GL/glew.h"

Cube::Cube(int param1) :
    cellsPerSide(param1),
    plane(new RectPlane(1, 1, cellsPerSide))
{

    plane->createTriangles();
    plane->translate(glm::vec3(-0.5, 0.f, -0.5));
    plane->rotate(M_PI, glm::vec3(1.f, 0.f, 0.f));
    plane->translate(glm::vec3(0, -0.5, 0));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI, glm::vec3(1.f, 0.f, 0.f));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI/2, glm::vec3(1.f, 0.f, 0.f));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI/2, glm::vec3(0.f, 1.f, 0.f));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI/2, glm::vec3(0.f, 1.f, 0.f));
    plane->addPoints(m_vertexData);
    plane->rotate(M_PI/2, glm::vec3(0.f, 1.f, 0.f));
    plane->addPoints(m_vertexData);
    buildVAO();
}

Cube::~Cube()
{
}
