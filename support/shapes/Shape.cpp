#include "Shape.h"

Shape::Shape(int param1, int param2) :
    m_param1(param1),
    m_param2(param2)
{
}

Shape::Shape() :
    m_param1(1),
    m_param2(1)
{
}



void Shape::setParam1(int param1) {
    m_param1 = param1;
    initializeVertexData();
    initializeOpenGLShapeProperties();
}

void Shape::setParam2(int param2) {
    m_param2 = param2;
    initializeVertexData();
    initializeOpenGLShapeProperties();
}

Shape::~Shape()
{
}
