#include "Surface.h"
#include <vector>
#include "GL/glew.h"
#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"

void insertVec3(std::vector<float> &data, glm::vec3 v){
    data.push_back(v.x);
    data.push_back(v.y);
    data.push_back(v.z);
}

Surface::Surface()
{
}

Surface::~Surface()
{
}

void Surface::addPoints(std::vector<GLfloat> &data){
    for (unsigned long i = 0; i < triangleBank.size(); i++)
    {
        Triangle *tri = &triangleBank[i];
        insertVec3(data, vertexBank[tri->a].position);
        insertVec3(data, vertexBank[tri->a].normal);
        insertVec3(data, vertexBank[tri->b].position);
        insertVec3(data, vertexBank[tri->b].normal);
        insertVec3(data, vertexBank[tri->c].position);
        insertVec3(data, vertexBank[tri->c].normal);
    }
}


void Surface::translate(glm::vec3 trans){
    translateVertexCollection(vertexBank, 0, vertexBank.size(), trans);
}


void Surface::translateVertex(Vertex* vert, glm::vec3 trans){
    glm::mat4 mat = glm::translate(trans);
    vert->position = glm::vec3(mat * glm::vec4(vert->position, 1.0));
}

void Surface::translateVertexCollection(std::vector<Vertex> &vec, unsigned long startIndex, unsigned long stopIndexExl, glm::vec3 trans){
    glm::mat4 mat = glm::translate(trans);
    for (unsigned long i = startIndex; i < stopIndexExl; i++)
    {
        Vertex v = vec[i];
        vec[i].position = glm::vec3(mat * glm::vec4(v.position, 1.0));
    }
}


void Surface::rotate(float angle, glm::vec3 axis){
    rotateVertexCollection(vertexBank, 0, vertexBank.size(), angle, axis);
}

void Surface::rotateVertex(Vertex* vert, float angle, glm::vec3 axis){
    glm::mat4 mat = glm::rotate(angle, axis);
    vert->position = glm::vec3(mat * glm::vec4(vert->position, 1.0));
    vert->normal = glm::vec3(mat * glm::vec4(vert->normal, 1.0));
}

void Surface::rotateVertexCollection(std::vector<Vertex> &vec, unsigned long startIndex, unsigned long stopIndexExl, float angle, glm::vec3 axis){
    glm::mat4 mat = glm::rotate(angle, axis);
    for (unsigned long i = startIndex; i < stopIndexExl; i++)
    {
        Vertex v = vec[i];
        vec[i].position = glm::vec3(mat * glm::vec4(v.position, 1.0));
        vec[i].normal = glm::vec3(mat * glm::vec4(v.normal, 1.0));
    }
}


void Surface::cleanup(){
    vertexBank.clear();
    triangleBank.clear();
}

