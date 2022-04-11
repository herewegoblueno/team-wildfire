#include "Texture2D.h"

#include <utility>

namespace CS123 { namespace GL {

Texture2D::Texture2D(unsigned char *data, int width, int height, GLenum type)
{
    GLenum internalFormat = type == GL_FLOAT ? GL_RGBA32F : GL_RGBA;

    // TODO [Task 2]
    // Bind the texture by calling bind() and filling it in
    // Generate the texture with glTexImage2D

    // TODO Don't forget to unbind!
}

void Texture2D::bind() const {
    // TODO [Task 2]
}

void Texture2D::unbind() const {
    // TODO Don't forget to unbind!
}

}}
