#ifndef SHAPE_H
#define SHAPE_H

#include "OpenGLShape.h"

/**
 *  Thin wrapper on OpenGLShape that allows setting param1 and param2.
 *  Includes a pure virtual method `initilializeVertexData` that all
 *  Shapes should implement to set their m_vertexData appopriately.
 */
class Shape : public OpenGLShape
{
public:
    Shape();
    Shape(int param1, int param2);
    ~Shape();
    void setParam1(int param1);
    void setParam2(int param2);

protected:
    int m_param1;
    int m_param2;
    virtual void initializeVertexData() = 0;
};

#endif // SHAPE_H
