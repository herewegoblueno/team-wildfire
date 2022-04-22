#include "CircleBase.h"
#include "TriMesh.h"
#include <math.h>
#include <iostream>
#include <string>
#include "glm/gtx/string_cast.hpp"
#include "glm/gtx/transform.hpp"

CircleBase::CircleBase(int param1, int param2, bool isTopBase) :
    Shape(param1, param2),
    m_tessellator(nullptr),
    m_isTopBase(isTopBase)
{
    m_tessellator = std::make_unique<Tessellator>();
    CircleBase::initializeVertexData();
    initializeOpenGLShapeProperties();
}

CircleBase::~CircleBase()
{
}


std::vector<glm::vec3> CircleBase::makeVertexGrid() {
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> grid;
    grid.reserve(height * width);

    float y;
    if (m_isTopBase) {
        y = 0.5f;
    } else {
        y = -0.5f;
    }

    for (int row = 0; row < height; row++) {
        float radius = 0.5f - (0.5f * static_cast<float>(row) / m_param1);
        for (int col = 0; col < width; col++) {
            float x;
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            // Reflect vertices if it's a top base so that we still get CCW order
            if (m_isTopBase) {
               x = radius * cos(theta);
            } else {
               x = -radius * cos(theta);
            }
            float z = radius * sin(theta);
            glm::vec3 vertex = glm::vec3(x, y, z);
            if (m_isTopBase && m_param2 % 2 > 0) {
                // If the base is asymmetric across the X-axis (i.e. param2 is odd),
                // we need to rotate the top base since we reflected it
                float rotationAngle =  PI / m_param2; // from 2 * PI / 2 * m_param2
                glm::mat4 rotate = glm::rotate(rotationAngle, glm::vec3(0.f, 1.0f, 0.f));
                vertex = glm::vec3(rotate * glm::vec4(vertex, 1.0f));
            }
            grid.push_back(vertex);
        }
    }
    return grid;
}

void CircleBase::initializeVertexData() {
    // Minimum param so that shape does not degenerate
    m_param2 = std::max(3, m_param2);
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> vertices = makeVertexGrid();
    std::vector<Triangle> faces = m_tessellator->tessellate(width, height);
    TriMesh triMesh = TriMesh(vertices, faces);
    removeDegenerateFaces(triMesh);
    m_tessellator->setUncurvedMeshNormals(triMesh);
    m_vertexData = m_tessellator->processTriMesh(triMesh);
    m_vertices = vertices;
    m_faces = faces;
}

/** Remove faces that have two of the same vertex */
void CircleBase::removeDegenerateFaces(TriMesh &triMesh) {
    std::vector<Triangle> newFaces;
    for (auto &face : triMesh.faces) {
        std::unordered_set<std::string> seen;
        bool degenerate = false;
        for (int i : face.vertexIndices) {
            std::string key = posToKey(triMesh.vertices[i]);
            if (seen.count(key)) {
                degenerate = true;
            }
            seen.insert(key);
        }
        if (!degenerate) {
            newFaces.push_back(face);
        }
    }
    triMesh.faces = newFaces;
}

std::string CircleBase::posToKey(glm::vec3 pos) {
    // add 0.0 to make -0.0 => 0.0
    std::string x = std::to_string(pos.x + 0.0);
    std::string y = std::to_string(pos.y + 0.0);
    std::string z = std::to_string(pos.z + 0.0);
    return x + "," + y + "," + z;
}

std::vector<float> CircleBase::getVertexData() {
    return m_vertexData;
}

std::vector<glm::vec3> CircleBase::getVertices() {
    return m_vertices;
}

std::vector<Triangle> CircleBase::getFaces() {
    return m_faces;
}


