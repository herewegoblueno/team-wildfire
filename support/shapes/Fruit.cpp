#include "Fruit.h"

Fruit::Fruit(int param1, int param2) :
    Shape(param1, param2)
{
    m_sphere = std::make_unique<Sphere>(m_param1, m_param2);
    Fruit::initializeVertexData();
    initializeOpenGLShapeProperties();
}

void Fruit::initializeVertexData() {
    m_vertexData = m_sphere->getVertexData();
}
