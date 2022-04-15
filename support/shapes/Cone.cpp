#include "Cone.h"
#include "TriMesh.h"
#include <math.h>
#include <iostream>
#include "glm/gtx/string_cast.hpp"

Cone::Cone(int param1, int param2) :
    Shape(param1, param2),
    m_tessellator(nullptr),
    m_bottom(nullptr)
{
    m_tessellator = std::make_unique<Tessellator>();
    m_bottom = std::make_unique<CircleBase>(m_param1, m_param2, false);
    Cone::initializeVertexData();
    initializeOpenGLShapeProperties();
}

Cone::~Cone()
{
}


std::vector<glm::vec3> Cone::makeSideGrid() {
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> grid;
    grid.reserve(height * width);
    for (int row = 0; row < height; row++) {
        // How far along the side of the cone we are, where 0 = top and 1 = base
        float t = static_cast<float>(row) / m_param1;
        float y = 0.5f - t;
        float radius = 0.5f * t;
        for (int col = 0; col < width; col++) {
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            float x = -radius * cos(theta);
            float z = radius * sin(theta);
            grid.push_back(glm::vec3(x, y, z));
        }
    }
    return grid;
}

void Cone::setSideNormals(std::vector<Triangle> &faces) {
    // We know the vertical/horizontal components of a normal vector for the cone
    float vertScalar = 1.0f / sqrt(5.0f);
    float horizScalar = 2.0f * vertScalar;
    int width = m_param2 + 1.0f;
    for (int i = 0; i < faces.size(); i++) {
        Triangle &face = faces[i];
        for (int j = 0; j < 3; j++) {
            int col = face.vertexIndices[j] % width;
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            float x = horizScalar * -cos(theta);
            float y = vertScalar;
            float z = horizScalar * sin(theta);
            face.vertexNormals.push_back(glm::vec3(x, y, z));
        }
    }
}

void Cone::initializeVertexData() {
    // Update params
    m_param2 = std::max(3, m_param2);
    m_bottom->setParam1(m_param1);
    m_bottom->setParam2(m_param2);

    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;

    // Build side of cone
    std::vector<glm::vec3> sideVertices = makeSideGrid();
    std::vector<Triangle> sideFaces = m_tessellator->tessellate(width, height);
    setSideNormals(sideFaces);
    TriMesh sideMesh = TriMesh(sideVertices, sideFaces);
    std::vector<float> sideVertexData = m_tessellator->processTriMesh(sideMesh);

    // Get bottom vertex data
    std::vector<float> bottomVertexData = m_bottom->getVertexData();

    // Combine vertex data
    sideVertexData.insert(sideVertexData.end(), bottomVertexData.begin(), bottomVertexData.end());
    m_vertexData = sideVertexData;
}
