#include "Torus.h"
#include <iostream>
#include <vector>
#include "GL/glew.h"

Torus::Torus(int param1, int param2, int param3) :
    sides(std::max(3, param1)),
    sideSmoothness(param2),
    thickness(std::max(1, param3/2)),
    radius(8.f)
{
    float theta = 2 * M_PI / (float)sides;
    float sideLength = sin(theta) * radius / sin(M_PI_2 - theta / 2);
    float sideIncline = M_PI_2 - theta / 2;
    float displacement = sin(sideIncline) * radius;
    loop = std::unique_ptr<Loop>(new Loop(sideLength, thickness, sideIncline, sideSmoothness, 1));
    loop->createTriangles();
    loop->translate(glm::vec3(displacement, 0, 0));
    for (int i = 0; i < sides; i++){
        loop->rotate(theta, glm::vec3(0, 0, 1));
        loop->addPoints(m_vertexData);
    }
    buildVAO();
}

Torus::~Torus()
{
}
