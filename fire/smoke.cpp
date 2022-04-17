#include "support/shapes/OpenGLShape.h"
#include "support/gl/textures/Texture2D.h"
#include "support/gl/textures/TextureParametersBuilder.h"

#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/FBO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"


#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"
#include <glm/gtx/transform.hpp>

#include "fire/smoke.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

Smoke::Smoke(int density, float frame_rate, float size):
    m_density(density*2), m_size(size), m_frame_rate(frame_rate)
{
    m_particles.clear();
    for (unsigned int i = 0; i < m_density; ++i)
        m_particles.push_back(Particle());

    InitRender();
}


void Smoke::InitRender()
{

    std::vector<float> particle_quad = { -1, 1, 0, 0, 0,
                               -1, -1, 0, 0, 1,
                               1, 1, 0, 1, 0,
                                1, -1, 0, 1, 1};

    m_quad = std::make_unique<OpenGLShape>();
    m_quad->m_vertexData = particle_quad;
    std::vector<CS123::GL::VBOAttribMarker> attribs;
    attribs.push_back(VBOAttribMarker(ShaderAttrib::POSITION, 3, 0, VBOAttribMarker::DATA_TYPE::FLOAT, false));
    attribs.push_back(VBOAttribMarker(ShaderAttrib::TEXCOORD0, 2, 3*sizeof(GLfloat), VBOAttribMarker::DATA_TYPE::FLOAT, false));
    CS123::GL::VBO vbo(particle_quad.data(), 20, attribs, VBO::GEOMETRY_LAYOUT::LAYOUT_TRIANGLE_STRIP);
    m_quad->m_VAO = std::make_unique<VAO>(vbo, 4);


    QImage img(":/textures/fire.png");
    QImage gl_img = QGLWidget::convertToGLFormat(img);
    m_texture = std::make_unique<Texture2D>(gl_img.bits(),
                                            gl_img.width(),
                                            gl_img.height());
}

Smoke::~Smoke()
{
//    glDeleteVertexArrays(1, &m_particlesVAO);
}


void Smoke::update_particles()
{
    // update all particles
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
        if (p.Life > 0.9f)
        {	// particle is alive, thus update
            p.Life += m_frame_rate; // reduce life
            p.Position += p.Velocity * m_frame_rate;
            p.Velocity.x = p.Velocity.x*1.01;
            p.Velocity.y = std::max(p.Velocity.y-0.001, 0.3);
            p.Velocity.z = p.Velocity.z*1.01;
        }
    }
}



void Smoke::RespawnParticle(int index, glm::vec3 pos, glm::vec3 vel)
{

    float vec_x = (rand() % 100 - 50)/ 120.0f * 0.2;
    float vec_z = (rand() % 100 - 50)/ 120.0f * 0.2;

    Particle &particle = m_particles[2*index];
    particle.Life = 1;
    particle.Position = pos;
    particle.Velocity = glm::vec3(vel.x-vec_x, 1.f - vel.y, vel.z-vec_z);
    particle.Color = glm::vec4(0.5, 0.5, 0.5, 1.0f);

    particle = m_particles[2*index+1];
    particle.Life = 1;
    particle.Position = pos;
    particle.Velocity = glm::vec3(vec_x*m_size, m_size*1.f - vel.y, vec_z*m_size);
    particle.Color = glm::vec4(0.5, 0.5, 0.5, 1.0f);
}

void Smoke::drawParticles(CS123::GL::CS123Shader* shader) {
    update_particles();

    for (Particle particle : m_particles)
    {
        if (particle.Life > 0.0f)
        {
            shader->setUniform("color", particle.Color);
            glm::mat4 M_fire = glm::translate(glm::mat4(), particle.Position);
            shader->setUniform("m", M_fire);
            shader->setUniform("temp", particle.Life);

            m_quad->draw();
        }
    }

}
