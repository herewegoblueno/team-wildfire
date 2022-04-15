#include "Leaf.h"


Leaf::Leaf() :
    Shape(),
    m_tessellator(nullptr)
{
    m_tessellator = std::make_unique<Tessellator>();
    Leaf::initializeVertexData();
    initializeOpenGLShapeProperties();
}

void Leaf::makeFrontVertexGrid() {
    glm::vec3 vertices[vertexRows * vertexCols] = {
        glm::vec3(0,0,0), glm::vec3(0.25,0.2,0), glm::vec3(0.5,0.25,0),
        glm::vec3(0.75,0.2,0), glm::vec3(1,0,0),
        glm::vec3(0,0,0), glm::vec3(0.25,0,0), glm::vec3(0.5,0,0),
        glm::vec3(0.75,0,0), glm::vec3(1,0,0),
        glm::vec3(0,0,0), glm::vec3(0.25,-0.2,0), glm::vec3(0.5,-0.25,0),
        glm::vec3(0.75,-0.2,0), glm::vec3(1,0,0),
    };
    m_frontVertexGrid.reserve(vertexRows * vertexCols);
    for (int i = 0; i < vertexRows * vertexCols; i++) {
        glm::vec3 vertex = vertices[i];
        m_frontVertexGrid.push_back(vertex);
    }
}

/**
 *  Reflect vertices to get "back" grid so that the leaf is visible
 *  from both sides with CCW tessellation
 */
void Leaf::makeBackVertexGrid() {
    m_backVertexGrid.reserve(vertexRows * vertexCols);
    for (int i = 0; i < vertexRows * vertexCols; i++) {
        glm::vec3 vertex = m_frontVertexGrid[i];
        m_backVertexGrid.push_back(glm::vec3(vertex.x, -vertex.y, vertex.z));
    }
}

void Leaf::initializeVertexData() {
    std::vector<Triangle> faces = m_tessellator->tessellate(vertexCols, vertexRows);

    makeFrontVertexGrid();
    TriMesh frontMesh = TriMesh(m_frontVertexGrid, faces);
    m_tessellator->setUncurvedMeshNormals(frontMesh);
    std::vector<float> frontVertexData = m_tessellator->processTriMesh(frontMesh);

    makeBackVertexGrid();
    TriMesh backMesh = TriMesh(m_backVertexGrid, faces);
    m_tessellator->setUncurvedMeshNormals(backMesh);
    std::vector<float> backVertexData = m_tessellator->processTriMesh(backMesh);

    frontVertexData.insert(frontVertexData.end(), backVertexData.begin(), backVertexData.end());
    m_vertexData = frontVertexData;
}
