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

#include "fire/fire.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

Fire::Fire(int density, glm::vec3 center, float size):
    m_density(density), m_size(size), m_center(center)
{
    // init particle
    m_particles.clear();
    for (unsigned int i = 0; i < density; ++i)
        m_particles.push_back(Particle());
    // init gaussian distribution
    dist = std::normal_distribution<float>(mean, stddev);
    //set respawn rate based on given density
    assert(m_density>2);
    m_respawn_num = fire_frame_rate * m_density / m_life;
    // init renderer
    InitRender();
    // init smoke
    m_smoke = std::make_unique<Smoke>(density, fire_frame_rate, size);
}


void Fire::InitRender()
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


    QImage img(":/textures/fire2.png");
    QImage gl_img = QGLWidget::convertToGLFormat(img);
    m_texture = std::make_unique<Texture2D>(gl_img.bits(),
                                            gl_img.width(),
                                            gl_img.height());
}

Fire::~Fire()
{
}


void Fire::update_particles()
{
    unsigned int nr_new_particles = m_respawn_num;
    // add new particles
    for (unsigned int i = 0; i < nr_new_particles; ++i)
    {
        int unusedParticle = FirstUnusedParticle();
        if(unusedParticle<m_density)
            RespawnParticle(m_particles[unusedParticle]);
    }
    // update all particles
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
        p.Life -= fire_frame_rate; // reduce life


        if (p.Life > 0.0f)
        {	// particle is alive, thus update
            p.Position += p.Velocity * fire_frame_rate;
            p.Velocity.x = p.Velocity.x*0.9;
            p.Velocity.y = std::max(p.Velocity.y-0.001, 0.1);
            p.Velocity.z = p.Velocity.z*0.9;
            p.Color.a -= fire_frame_rate * 2.5f;
        }


        if(p.Life < fire_frame_rate*1.5)
        {
            m_smoke->RespawnParticle(i, p.Position, p.Velocity);
        }
    }
}



void Fire::RespawnParticle(Particle &particle)
{

    float random_x = dist(generator) / 10.0f*m_size;
//    float random_y = dist(generator) / 20.0f*m_size;
    float random_y = 0;
//    float random_y = ((rand() % 100) - 100) / 500.0f*m_size;
    float random_z = dist(generator) / 10.0f*m_size;
    glm::vec3 offset(random_x, random_y, random_z);
    offset = offset*3.f;

    float rColor = 0.5f + ((rand() % 50) / 100.0f);
    particle.Position = m_center + offset;
    particle.Color = glm::vec4(rColor, rColor, rColor, 1.0f);
    particle.Life = m_life;

    float vec_y = (rand() % 50)/ 50.0f + 0.2;
    float vec_x = (rand() % 100 - 50)/ 120.0f * (1-vec_y)*m_size;
    float vec_z = (rand() % 100 - 50)/ 120.0f * (1-vec_y)*m_size;
    particle.Velocity = glm::vec3(vec_x, vec_y*0.3f+0.1 , vec_z);
}



void Fire::drawSmoke(CS123::GL::CS123Shader* shader) {
    m_smoke->drawParticles(shader);
}

void Fire::drawParticles(CS123::GL::CS123Shader* shader) {
    update_particles();


//    // bind texture
    TextureParametersBuilder builder;
    builder.setFilter(TextureParameters::FILTER_METHOD::LINEAR);
    builder.setWrap(TextureParameters::WRAP_METHOD::REPEAT);

    TextureParameters parameters = builder.build();
    parameters.applyTo(*m_texture);
    std::string filename = "fire2.png";
    shader->setTexture(filename, *m_texture);

    for (Particle particle : m_particles)
    {
        if (particle.Life > 0.0f)
        {
            shader->setUniform("color", particle.Color);
            glm::mat4 M_fire = glm::translate(glm::mat4(), particle.Position);
            shader->setUniform("m", M_fire);
            shader->setUniform("life", particle.Life);

            m_quad->draw();
        }
    }

}


unsigned int Fire::FirstUnusedParticle()
{
    // search from last used particle, this will usually return almost instantly
    for (unsigned int i = lastUsedParticle; i < m_density; ++i) {
        if (m_particles[i].Life <= 0.0f){
            lastUsedParticle = i;
            return i;
        }
    }
    // otherwise, do a linear search
    for (unsigned int i = 0; i < lastUsedParticle; ++i) {
        if (m_particles[i].Life <= 0.0f){
            lastUsedParticle = i;
            return i;
        }
    }
    // override first particle if all others are alive
    lastUsedParticle = 0;
    return 0;
}
