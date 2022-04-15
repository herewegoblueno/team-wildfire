#include "Cube.h"
#include "TriMesh.h"
#include <iostream>

Cube::Cube(int param1, int param2) :
    Shape(param1, param2),
    m_tessellator(nullptr)
{
    m_tessellator = std::make_unique<Tessellator>();
    Cube::initializeVertexData();
    initializeOpenGLShapeProperties();
}

Cube::~Cube()
{
}

/** Get vertex grid for the top of the cube */
std::vector<glm::vec3> Cube::makeTopGrid() {
    int sideLength = m_param1 + 1;
    std::vector<glm::vec3> grid;
    grid.reserve(sideLength * sideLength);
    float y = 0.5f;
    for (int row = 0; row < sideLength; row++) {
        for (int col = 0; col < sideLength; col++) {
            float x = -0.5f + static_cast<float>(col) / m_param1;
            float z = -0.5f + static_cast<float>(row) / m_param1;
            grid.push_back(glm::vec3(x, y, z));
        }
    }
    return grid;
}

/** Get vertex grid for the bottom of the cube */
std::vector<glm::vec3> Cube::makeBottomGrid() {
    int sideLength = m_param1 + 1;
    std::vector<glm::vec3> grid;
    grid.reserve(sideLength * sideLength);
    float y = -0.5f;
    for (int row = 0; row < sideLength; row++) {
        for (int col = 0; col < sideLength; col++) {
            // Since we tessellate from a viewpoint "above" the object, we need to
            // reflect the vertex grid for the bottom in order to get CCW order
            float x = 0.5f - static_cast<float>(col) / m_param1;
            float z = -0.5f + static_cast<float>(row) / m_param1;
            grid.push_back(glm::vec3(x, y, z));
        }
    }
    return grid;
}


/** Get vertex grid for the side of the cube */
std::vector<glm::vec3> Cube::makeSideGrid() {
    // Number of y values
    int height = m_param1 + 1;
    // Number of t values, where t is in [0, 4] and determines
    // how far along the square in the XZ plane the vertex is
    // (We parameterize the square into 4 line segments)
    int width = 4 * m_param1 + 1;
    std::vector<glm::vec3> sideGrid;
    sideGrid.reserve(height * width);
    for (int row = 0; row < height; row++) {
        float y = 0.5f - (static_cast<float>(row) / m_param1);
        for (int col = 0; col < width; col++) {
            float x, z;
            float t = 4.0f * static_cast<float>(col) / (width - 1.0f);
            if (t <= 1) {
                x = -0.5f;
                z = -0.5f + t;
            } else if (t <= 2) {
                x = -0.5f + t - 1;
                z = 0.5f;
            } else if (t <= 3) {
                x = 0.5f;
                z = 0.5f - (t - 2);
            } else if (t <= 4) {
                x = 0.5f - (t - 3);
                z = -0.5f;
            } else {
                std::cout << "CUBE t-value invalid (> 4)" << std::endl;
            }
            sideGrid.push_back(glm::vec3(x, y, z));
        }
    }
    return sideGrid;
}

void Cube::initializeVertexData() {
    // Number of y values
    int height = m_param1 + 1;
    // Number of t values, where t is in [0, 4] and determines
    // how far along the square in the XZ plane the vertex is
    // (We parameterize the square into 4 line segments)
    int width = 4 * m_param1 + 1;

    std::vector<glm::vec3> sideVertices = makeSideGrid();
    std::vector<glm::vec3> topVertices = makeTopGrid();
    std::vector<glm::vec3> bottomVertices = makeBottomGrid();

    std::vector<Triangle> sideFaces = m_tessellator->tessellate(width, height);
    std::vector<Triangle> topFaces = m_tessellator->tessellate(height, height);
    std::vector<Triangle> bottomFaces = m_tessellator->tessellate(height, height);

    TriMesh sideMesh = TriMesh(sideVertices, sideFaces);
    TriMesh topMesh = TriMesh(topVertices, topFaces);
    TriMesh bottomMesh = TriMesh(bottomVertices, bottomFaces);

    m_tessellator->setUncurvedMeshNormals(sideMesh);
    m_tessellator->setUncurvedMeshNormals(topMesh);
    m_tessellator->setUncurvedMeshNormals(bottomMesh);

    std::vector<float> sideVertexData = m_tessellator->processTriMesh(sideMesh);
    std::vector<float> topVertexData = m_tessellator->processTriMesh(topMesh);
    std::vector<float> bottomVertexData = m_tessellator->processTriMesh(bottomMesh);

    sideVertexData.insert(sideVertexData.end(), topVertexData.begin(), topVertexData.end());
    sideVertexData.insert(sideVertexData.end(), bottomVertexData.begin(), bottomVertexData.end());

    m_vertexData = sideVertexData;
}
