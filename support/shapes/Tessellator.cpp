#include "Tessellator.h"
#include "TriMesh.h"
#include <iostream>

Tessellator::Tessellator() {

}

Tessellator::~Tessellator()
{
}

/**
 *  Take completed triMesh and convert it into the vertex data format
 *  that we can pass to OpenGL.
 */
std::vector<float> Tessellator::processTriMesh(TriMesh triMesh) {
    std::vector<Triangle> faces = triMesh.faces;
    std::vector<glm::vec3> vertices = triMesh.vertices;
    int numFaces = faces.size();
    std::vector<float> vertexData;
    // Each face has 3 vertices, which each need 2 vectors (vertex, normal), which each have 3 floats
    vertexData.reserve(numFaces * 3 * 2 * 3);
    // Insert each vertex followed by it's face normal
    for (int i = 0; i < numFaces; i++) {
        Triangle face = faces[i];
        for (int j = 0; j < 3; j++) {
            glm::vec3 vertex = vertices[face.vertexIndices[j]];
            glm::vec3 normal = face.vertexNormals.at(j);
            insertVec3(vertexData, vertex);
            insertVec3(vertexData, normal);
        }
    }
    return vertexData;
}

/**
 *  Tessellate vertices into counter-clockwise-ordered triangles.
 *  Takes in width and height of vertex 2D array. Since a Triangle is
 *  just 3 vertex indices, this method does not need to take in actual
 *  glm::vec3 vertices.
 */
std::vector<Triangle> Tessellator::tessellate(int gridWidth, int gridHeight) {
    std::vector<Triangle> faces;
    faces.reserve(gridWidth * gridHeight);
    // Whether we are currently making a top-left triangle or a bottom-right triangle (alternates)
    bool topLeftTriangle = true;
    // Our triangles extend "down" from the start vertex, so we stop before the last row
    for (int row = 0; row < gridHeight - 1; row++) {
        int col = 0;
        // We use the counter to make we only increment col every 2 iterations
        int counter = 0;
        while (col < gridWidth) {
            int vertex0 = row * gridWidth + col;
            int vertex1, vertex2;
            if (topLeftTriangle) {
                // For top-left triangles, we don't want to go past the right edge of the grid
                if (col == gridWidth - 1) {
                    if (counter % 2 == 0) {
                        col++;
                    }
                    counter++;
                    continue;
                }
                // Down 1
                vertex1 = (row + 1) * gridWidth + col;
                // Right 1
                vertex2 = row * gridWidth + col + 1;
            } else {
                // Down 1 Left 1
                vertex1 = (row + 1) * gridWidth + col - 1;
                // Down 1
                vertex2 = (row + 1) * gridWidth + col;
            }
            faces.push_back(Triangle(vertex0, vertex1, vertex2));
            topLeftTriangle = !topLeftTriangle;
            if (counter % 2 == 0) {
                col++;
            }
            counter++;
        }
    }
    return faces;
}

/**
 *  For meshes where all vertices for a given face have the same normal,
 *  compute and set the normal using crossproducts
 */
void Tessellator::setUncurvedMeshNormals(TriMesh triMesh) {
    std::vector<glm::vec3> &vertices = triMesh.vertices;
    std::vector<Triangle> &faces = triMesh.faces;
    for (int i = 0; i < faces.size(); i++ ) {
        // Get vertex indices from face
        Triangle &face = faces[i];
        int vertexIndex0 = face.vertexIndices[0];
        int vertexIndex1 = face.vertexIndices[1];
        int vertexIndex2 = face.vertexIndices[2];
        // Access vertices
        glm::vec3 vertex0 = vertices[vertexIndex0];
        glm::vec3 vertex1 = vertices[vertexIndex1];
        glm::vec3 vertex2 = vertices[vertexIndex2];
        // Compute normal by using the normalized cross-product of the triangle vectors
        glm::vec3 vectorA = vertex1 - vertex0;
        glm::vec3 vectorB = vertex2 - vertex0;
        glm::vec3 normal = glm::normalize(glm::cross(vectorA, vectorB));
        for (int i = 0; i < 3; i++) {
            face.vertexNormals.push_back(normal);
        }
    }
}
