#include "Texture.h"

#include <cassert>
#include <utility>

#include <GL/glew.h>
#include "support/gl/GLDebug.h"

namespace CS123 { namespace GL {

Texture::Texture() :
    m_handle(0)
{
    // TODO [Task 2] Generate the texture
}

Texture::Texture(Texture &&that) :
    m_handle(that.m_handle)
{
    that.m_handle = 0;
}

Texture& Texture::operator=(Texture &&that) {
    this->~Texture();
    m_handle = that.m_handle;
    that.m_handle = 0;
    return *this;
}

Texture::~Texture()
{
    // TODO Don't forget to delete!
}

unsigned int Texture::id() const {
    return m_handle;
}

}}
