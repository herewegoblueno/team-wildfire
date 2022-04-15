#ifndef TERRAIN_H
#define TERRAIN_H

#include "glm/glm.hpp"            // glm::vec*, mat*, and basic glm functions
#include "glm/gtx/transform.hpp"  // glm::translate, scale, rotate
#include "glm/gtc/type_ptr.hpp"   // glm::value_ptr
#include <vector>

#include "support/shapes/OpenGLShape.h"
#include "memory"

const int scale = 40;

class Terrain {
public:
    Terrain();

    std::vector<float> init();
    void draw();

    std::unique_ptr<OpenGLShape> openGLShape;
    bool isFilledIn();

    float getHeightFromWorld(glm::vec3 pos);
    glm::vec3 getNormalFromWorld(glm::vec3 pos);

private:
    float randValue(int row, int col);
    glm::vec3 getPosition(int row, int col);
    glm::vec3 getNormal(int row, int col);
    const float m_numRows, m_numCols;
    const bool m_isFilledIn;
};

#endif // TERRAIN_H
