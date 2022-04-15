#include "Sphere.h"
#include "TriMesh.h"
#include <math.h>
#include <iostream>
#include "glm/gtx/string_cast.hpp"
#include "glm/gtx/transform.hpp"

Sphere::Sphere(int param1, int param2) :
    Shape(param1, param2),
    m_tessellator(nullptr)
{
    m_tessellator = std::make_unique<Tessellator>();
    Sphere::initializeVertexData();
    initializeOpenGLShapeProperties();
}

Sphere::~Sphere()
{
}


std::vector<glm::vec3> Sphere::makeVertexGrid() {
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> grid;
    grid.reserve(height * width);
    for (int row = 0; row < height; row++) {
        // the vertical spherical angle, where 0 => top and PI => bottom of sphere
        float phi = PI - (PI * static_cast<float>(row) / m_param1);
        float y = 0.5 * cos(phi);
        for (int col = 0; col < width; col++) {
            float theta = 2.0f * PI * static_cast<float>(col) / m_param2;
            float x = 0.5 * sin(phi) * cos(theta);
            float z = 0.5 * sin(phi) * sin(theta);
            grid.push_back(glm::vec3(x, y , z));
        }
    }
    return grid;
}

void Sphere::setSphereNormals(TriMesh triMesh) {
    std::vector<glm::vec3> &vertices = triMesh.vertices;
    std::vector<Triangle> &faces = triMesh.faces;
    for (int i = 0; i < faces.size(); i++ ) {
        Triangle &face = faces[i];
        for (int j = 0; j < 3; j++) {
            int vertexIndex = face.vertexIndices[j];
            glm::vec3 vertex = vertices[vertexIndex];
            glm::vec3 normal = glm::normalize(vertex);
            face.vertexNormals.push_back(normal);
        }
    }
}

void Sphere::initializeVertexData() {
    // Minimum params so that shape does not degenerate
    m_param1 = std::max(2, m_param1);
    m_param2 = std::max(3, m_param2);
    int height = m_param1 + 1.0f;
    int width = m_param2 + 1.0f;
    std::vector<glm::vec3> vertices = makeVertexGrid();
    std::vector<Triangle> faces = m_tessellator->tessellate(width, height);
    TriMesh triMesh = TriMesh(vertices, faces);
    setSphereNormals(triMesh);
    m_vertexData = m_tessellator->processTriMesh(triMesh);
}

std::vector<float> Sphere::getVertexData() {
    return m_vertexData;
}
