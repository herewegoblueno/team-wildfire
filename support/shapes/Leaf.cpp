#include "Leaf.h"

Leaf::Leaf()
{
    Leaf::initializeVertexData();
    initializeOpenGLShapeProperties();
}

void Leaf::initializeVertexData() {
    buildFront();
    buildBack();
}

void Leaf::buildFront() {
    std::vector<glm::vec3> vertices = {
        glm::vec3(0, 0, 0),
        glm::vec3(0.5, -0.25, 0),
        glm::vec3(0.5, 0.25, 0),
        glm::vec3(0.5, -0.25, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(0.5, 0.25, 0),
    };
    for (auto &vertex : vertices) {
        insertVec3(m_vertexData, vertex);
        insertVec3(m_vertexData, glm::vec3(0, 0, 1)); // normal
    }
}

void Leaf::buildBack() {
    std::vector<glm::vec3> vertices = {
        glm::vec3(0, 0, 0),
        glm::vec3(0.5, 0.25, 0),
        glm::vec3(0.5, -0.25, 0),
        glm::vec3(0.5, 0.25, 0),
        glm::vec3(1, 0, 0),
        glm::vec3(0.5, -0.25, 0),
    };
    for (auto &vertex : vertices) {
        insertVec3(m_vertexData, vertex);
        insertVec3(m_vertexData, glm::vec3(0, 0, -1)); // normal
    }
}
