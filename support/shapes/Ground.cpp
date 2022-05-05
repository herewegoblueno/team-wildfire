#include "Ground.h"

/** Square unit ground plane made of four triangles, centered at origin */
Ground::Ground()
{
    Ground::initializeVertexData();
    initializeOpenGLShapeProperties();
}

void Ground::initializeVertexData() {
    buildTop();
    buildBottom();
}

void Ground::buildTop() {
    std::vector<glm::vec3> vertices = {
        glm::vec3(-0.5, 0, -0.5),
        glm::vec3(0.5, 0, 0.5),
        glm::vec3(-0.5, 0, 0.5),
        glm::vec3(-0.5, 0, -0.5),
        glm::vec3(0.5, 0, -0.5),
        glm::vec3(0.5 ,0, 0.5),
    };
    for (auto &vertex : vertices) {
        insertVec3(m_vertexData, vertex);
        insertVec3(m_vertexData, glm::vec3(0, 1, 0)); // normal
    }
}

void Ground::buildBottom() {
    std::vector<glm::vec3> vertices = {
        glm::vec3(-0.5, 0, 0.5),
        glm::vec3(0.5, 0, 0.5),
        glm::vec3(-0.5, 0, -0.5),
        glm::vec3(0.5, 0, 0.5),
        glm::vec3(0.5, 0, -0.5),
        glm::vec3(-0.5, 0, -0.5),
    };
    for (auto &vertex : vertices) {
        insertVec3(m_vertexData, vertex);
        insertVec3(m_vertexData, glm::vec3(0, -1, 0)); // normal
    }
}
