#include "OpenGLShape.h"
#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"

using namespace CS123::GL;

OpenGLShape::OpenGLShape() :
    m_VAO(nullptr),
    needs_triangle_strip(false)
{

}

OpenGLShape::OpenGLShape(bool t_strip) :
    m_VAO(nullptr),
    needs_triangle_strip(t_strip)
{

}

OpenGLShape::~OpenGLShape()
{
}

void OpenGLShape::draw() {
    if (m_VAO) {
        m_VAO->bind();
        m_VAO->draw();
        m_VAO->unbind();
    }
}

void OpenGLShape::setVertexData()
{

}


void OpenGLShape::buildVAO()
{

}


void OpenGLShape::initializeOpenGLShapeProperties() {
    const int numFloatsPerVertex = 6;
    const int numVertices = m_vertexData.size() / numFloatsPerVertex;

    std::vector<VBOAttribMarker> markers;
    markers.push_back(VBOAttribMarker(ShaderAttrib::POSITION, 3, 0));
    markers.push_back(VBOAttribMarker(ShaderAttrib::NORMAL, 3, 3*sizeof(float)));
    VBO vbo1 = VBO(m_vertexData.data(), m_vertexData.size(), markers);
    VBO vbo2 = VBO(m_vertexData.data(), m_vertexData.size(), markers, VBO::GEOMETRY_LAYOUT::LAYOUT_TRIANGLE_STRIP);
    m_VAO = std::make_unique<VAO>(needs_triangle_strip ? vbo2 : vbo1, numVertices);
}
