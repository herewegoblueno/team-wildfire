#ifndef SURFACE_H
#define SURFACE_H

#include <glm/glm.hpp>
#include <vector>
#include "GL/glew.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

struct Triangle { //Refers to indexes in vertexbank
  int a;
  int b;
  int c;
};

void insertVec3(std::vector<float> &data, glm::vec3 v);


class Surface
{
public:
    Surface();
    virtual ~Surface();

    void addPoints(std::vector<GLfloat> &data);
    void rotate(float angle, glm::vec3 axis);
    void translate(glm::vec3 trans);
    void cleanup();
    virtual void createTriangles() = 0;

    std::vector<Vertex> vertexBank;
    std::vector<Triangle> triangleBank;

protected:
    void rotateVertex(Vertex *vert, float angle, glm::vec3 axis);
    void rotateVertexCollection(std::vector<Vertex> &vec, unsigned long startIndex, unsigned long stopIndex, float angle, glm::vec3 axis);
    void translateVertex(Vertex *vert, glm::vec3 trans);
    void translateVertexCollection(std::vector<Vertex> &vec, unsigned long startIndex, unsigned long stopIndex, glm::vec3 trans);
};

#endif // SURFACE_H
