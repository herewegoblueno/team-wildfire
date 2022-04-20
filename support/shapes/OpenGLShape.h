#ifndef OPENGLSHAPE_H
#define OPENGLSHAPE_H

/** imports the OpenGL math library https://glm.g-truc.net/0.9.2/api/a00001.html */
#include <glm/glm.hpp>
#include "GL/glew.h"

#include<memory>
#include <vector>

/**
 *
 * inserts a glm::vec3 into a vector of floats
 * this will come in handy if you want to take advantage of vectors to build your shape
 * make sure to call reserve beforehand to speed this up
 */
inline void insertVec3(std::vector<float> &data, glm::vec3 v){
    data.push_back(v.x);
    data.push_back(v.y);
    data.push_back(v.z);
}

namespace CS123 { namespace GL {
class VAO;
}}

class OpenGLShape
{
public:
    OpenGLShape();
    OpenGLShape(bool t_strip);
    virtual ~OpenGLShape();
    void draw();

    void setVertexData();
    void buildVAO();
    void bindVAO();
    void drawVAO();
    void unbindVAO();

    /**
     * initializes the relavant openGL properties for the shape
     * don't worry about what exactly this function is doing, you'll learn more about that later in the course!
     * look at ExampleShape.cpp for it's demonstrated usage
     */
    void initializeOpenGLShapeProperties();

    std::vector<GLfloat> m_vertexData;
    std::unique_ptr<CS123::GL::VAO> m_VAO;
    bool needs_triangle_strip;
};

#endif // OPENGLSHAPE_H
