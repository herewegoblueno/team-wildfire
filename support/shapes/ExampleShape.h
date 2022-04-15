#ifndef EXAMPLESHAPE_H
#define EXAMPLESHAPE_H

#include "OpenGLShape.h"

// hand-written cube points and normals.. if only there were a way to do this procedurally
#define CUBE_DATA_POSITIONS {\
        -0.5f,-0.5f,-0.5f, \
        -1.f, 0.f, 0.f, \
        -0.5f,-0.5f, 0.5f,\
        -1.f, 0.f, 0.f, \
        -0.5f, 0.5f, 0.5f, \
        -1.f, 0.f, 0.f, \
        0.5f, 0.5f,-0.5f, \
        0.f, 0.f, -1.f, \
        -0.5f,-0.5f,-0.5f,\
        0.f, 0.f, -1.f, \
        -0.5f, 0.5f,-0.5f, \
        0.f, 0.f, -1.f, \
        0.5f,-0.5f, 0.5f, \
        0.f, -1.f, 0.f, \
        -0.5f,-0.5f,-0.5f, \
        0.f, -1.f, 0.f, \
        0.5f,-0.5f,-0.5f, \
        0.f, -1.f, 0.f, \
        0.5f, 0.5f,-0.5f, \
        0.f, 0.f, -1.f, \
        0.5f,-0.5f,-0.5f, \
        0.f, 0.f, -1.f, \
        -0.5f,-0.5f,-0.5f, \
        0.f, 0.f, -1.f, \
        -0.5f,-0.5f,-0.5f, \
        -1.f, 0.f, 0.f, \
        -0.5f, 0.5f, 0.5f,\
        -1.f, 0.f, 0.f, \
        -0.5f, 0.5f,-0.5f,\
        -1.f, 0.f, 0.f, \
        0.5f,-0.5f, 0.5f,\
        0.f, -1.f, 0.f, \
        -0.5f,-0.5f, 0.5f,\
        0.f, -1.f, 0.f, \
        -0.5f,-0.5f,-0.5f,\
        0.f, -1.f, 0.f, \
        -0.5f, 0.5f, 0.5f,\
        0.f, 0.f, 1.f, \
        -0.5f,-0.5f, 0.5f,\
        0.f, 0.f, 1.f, \
        0.5f,-0.5f, 0.5f,\
        0.f, 0.f, 1.f, \
        0.5f, 0.5f, 0.5f,\
        1.f, 0.f, 0.f, \
        0.5f,-0.5f,-0.5f,\
        1.f, 0.f, 0.f, \
        0.5f, 0.5f,-0.5f,\
        1.f, 0.f, 0.f, \
        0.5f,-0.5f,-0.5f,\
        1.f, 0.f, 0.f, \
        0.5f, 0.5f, 0.5f,\
        1.f, 0.f, 0.f, \
        0.5f,-0.5f, 0.5f,\
        1.f, 0.f, 0.f, \
        0.5f, 0.5f, 0.5f,\
        0.f, 1.f, 0.f, \
        0.5f, 0.5f,-0.5f,\
        0.f, 1.f, 0.f, \
        -0.5f, 0.5f,-0.5f,\
        0.f, 1.f, 0.f, \
        0.5f, 0.5f, 0.5f,\
        0.f, 1.f, 0.f, \
        -0.5f, 0.5f,-0.5f,\
        0.f, 1.f, 0.f, \
        -0.5f, 0.5f, 0.5f,\
        0.f, 1.f, 0.f, \
        0.5f, 0.5f, 0.5f,\
        0.f, 0.f, 1.f, \
        -0.5f, 0.5f, 0.5f,\
        0.f, 0.f, 1.f, \
        0.5f,-0.5f, 0.5f, \
        0.f, 0.f, 1.f}

class ExampleShape : public OpenGLShape
{
public:
    ExampleShape();
    ExampleShape(int param1, int param2);
    ~ExampleShape();

private:
    int m_param1;
    int m_param2;
};

#endif // EXAMPLESHAPE_H
