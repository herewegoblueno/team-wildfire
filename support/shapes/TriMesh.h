#ifndef TRIMESH_H
#define TRIMESH_H
#include <glm/glm.hpp>
#include <memory>
#include <vector>

/**
 *  A triangle face is defined via the indices and normals of its vertices.
 */
struct Triangle {
    std::vector<int> vertexIndices;
    std::vector<glm::vec3> vertexNormals;

    Triangle(int vert0, int vert1, int vert2) :
        vertexIndices({vert0, vert1, vert2})
    {
        vertexNormals.reserve(3);
    }
};

/**
 *  A triangle mesh is defined via reference to a list of vertices and a list of triangle faces
 */
struct TriMesh {
    std::vector<glm::vec3> &vertices;
    std::vector<Triangle> &faces;

    TriMesh(std::vector<glm::vec3> &vertices, std::vector<Triangle> &faces) :
          vertices(vertices),
          faces(faces)
    {
    }
};

#endif // TRIMESH_H
