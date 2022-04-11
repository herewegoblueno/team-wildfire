#include "FullScreenQuad.h"

#include <vector>

#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/datatype/VAO.h"
#include "support/gl/shaders/ShaderAttribLocations.h"

using namespace CS123::GL;

FullScreenQuad::FullScreenQuad()
{
    float data[] = { -1.0f,  1.0f,   0.0f, 1.0f,
                     -1.0f, -1.0f,   0.0f, 0.0f,
                      1.0f,  1.0f,   1.0f, 1.0f,
                      1.0f, -1.0f,   1.0f, 0.0f };

    std::vector<CS123::GL::VBOAttribMarker> attribs;
    attribs.push_back(VBOAttribMarker(ShaderAttrib::POSITION, 2, 0));
    attribs.push_back(VBOAttribMarker(ShaderAttrib::TEXCOORD0, 2, 2*sizeof(float)));
    VBO vbo(data, sizeof(data), attribs, VBO::GEOMETRY_LAYOUT::LAYOUT_TRIANGLE_STRIP);
    m_vao = std::make_unique<VAO>(vbo, 4);
}

void FullScreenQuad::draw() {
    m_vao->bind();
    m_vao->draw();
    m_vao->unbind();
}
