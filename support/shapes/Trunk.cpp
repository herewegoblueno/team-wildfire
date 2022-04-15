#include "Trunk.h"
#include "TriMesh.h"
#include <math.h>
#include <iostream>
#include "glm/gtx/string_cast.hpp"

Trunk::Trunk(int param1, int param2) :
    Shape(param1, param2),
    m_taperAmt(1.f - branchWidthDecay),
    m_tessellator(nullptr),
    m_top(nullptr),
    m_bottom(nullptr)
{
    m_tessellator = std::make_unique<Tessellator>();
    m_top = std::make_unique<CircleBase>(m_param1, m_param2, true);
    m_bottom = std::make_unique<CircleBase>(m_param1, m_param2, false);
    Trunk::initializeVertexData();
    initializeOpenGLShapeProperties();
}

Trunk::~Trunk()
{
}


std::vector<glm::vec3> Trunk::makeSideGrid() {
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> grid;
    grid.reserve(height * width);
    for (int row = 0; row < height; row++) {
        float y = 0.5f - static_cast<float>(row) / m_param1;
        float horizScale = 1.f - m_taperAmt * (y + 0.5f) / 1.0f;
        for (int col = 0; col < width; col++) {
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            float x = -0.5f * cos(theta) * horizScale;
            float z = 0.5f * sin(theta) * horizScale;
            grid.push_back(glm::vec3(x, y, z));
        }
    }
    return grid;
}

void Trunk::setSideNormals(std::vector<Triangle> &faces) {
    float slope = 1.f / m_taperAmt;
    int width = m_param2 + 1.0f;
    for (int i = 0; i < faces.size(); i++) {
        Triangle &face = faces[i];
        for (int j = 0; j < 3; j++) {
            int col = face.vertexIndices[j] % width;
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            float x = slope * -cos(theta);
            float y = 1;
            float z = slope * sin(theta);
            glm::vec3 normal = glm::normalize(glm::vec3(x, y, z));
            face.vertexNormals.push_back(normal);
        }
    }
}

void Trunk::initializeVertexData() {
    // Update params
    m_param2 = std::max(3, m_param2);
    m_top->setParam1(m_param1);
    m_top->setParam2(m_param2);
    m_bottom->setParam1(m_param1);
    m_bottom->setParam2(m_param2);

    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;

    // Build side of cylinder
    std::vector<glm::vec3> sideVertices = makeSideGrid();
    std::vector<Triangle> sideFaces = m_tessellator->tessellate(width, height);
    setSideNormals(sideFaces);
    TriMesh sideMesh = TriMesh(sideVertices, sideFaces);
    std::vector<float> sideVertexData = m_tessellator->processTriMesh(sideMesh);

    // Get bottom vertex data
    std::vector<float> bottomVertexData = m_bottom->getVertexData();

    // Get top vertices and scale based on taper
    std::vector<glm::vec3> topVertices = m_top->getVertices();
    topVertices = scaleTop(topVertices);
    std::vector<Triangle> topFaces = m_top->getFaces();
    TriMesh topMesh = TriMesh(topVertices, topFaces);
    m_tessellator->setUncurvedMeshNormals(topMesh);
    std::vector<float> topVertexData = m_tessellator->processTriMesh(topMesh);

    // Combine vertex data
    sideVertexData.insert(sideVertexData.end(), topVertexData.begin(), topVertexData.end());
    sideVertexData.insert(sideVertexData.end(), bottomVertexData.begin(), bottomVertexData.end());
    m_vertexData = sideVertexData;
}

/** Scale the top of the tapered cylinder to match the tapered sides */
std::vector<glm::vec3> Trunk::scaleTop(std::vector<glm::vec3> vertices) {
    std::vector<glm::vec3> newVertices;
    int numVertices = vertices.size();
    newVertices.reserve(numVertices);
    for (int i = 0; i < numVertices; i++) {
        glm::vec3 vert = vertices[i];
        newVertices.push_back(
                    glm::vec3(branchWidthDecay * vert.x, vert.y, branchWidthDecay * vert.z));
    }
    return newVertices;
}
