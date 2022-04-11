#ifndef SHAPE_H
#define SHAPE_H

/** imports the OpenGL math library https://glm.g-truc.net/0.9.2/api/a00001.html */
#include <glm/glm.hpp>
#include "GL/glew.h"

#include<memory>
#include <vector>

namespace CS123 { namespace GL {
class VAO;
}}

class Shape
{
public:
    Shape();
    ~Shape();
    void draw();

protected:
    /** builds the VAO, pretty much the same as from lab 1 */
    void buildVAO();

    std::vector<GLfloat> m_vertexData;
    std::unique_ptr<CS123::GL::VAO> m_VAO;
};

#endif // SHAPE_H
