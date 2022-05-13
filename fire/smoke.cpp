#include "support/shapes/OpenGLShape.h"

#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/FBO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"


#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"
#include <glm/gtx/transform.hpp>

#include "fire/smoke.h"
#include "voxels/voxelgrid.h"
#include "simulation/physics.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

Smoke::Smoke(int density, float frame_rate, float size, VoxelGrid* grid):
    m_density(density*2), m_size(size), m_grid(grid)
{
    m_particles.clear();
    m_offset.clear();
    dist = std::normal_distribution<float>(mean, stddev);
    for (unsigned int i = 0; i < m_density*2; ++i)
    {
        m_particles.push_back(Particle());
        float random_x = dist(generator);
        float random_y = dist(generator);

        m_offset.push_back(glm::vec2(random_x, random_y));
    }
}



Smoke::~Smoke()
{

}


void Smoke::update_particles(float timeStep)
{
    if (timeStep == 0) return;
    // update all particles
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
        if (p.Life > 0.4f)
        {	// particle is alive, thus update
            VoxelPhysicalData voxel = m_grid->getStateInterpolatePoint(p.Position);
            VoxelPhysicalData* vox = &voxel;
            float ambient_T = vox->temperature;
            if(std::isnan(ambient_T)) ambient_T = p.Temp;
            glm::vec3 u = vec3(vox->u)*1000.f;
            u.y = std::max(u.y, 0.f);

            float b_factor = 0.3;
            #ifdef CUDA_FLUID
                b_factor = 0.2;
            #endif

            float b = gravity_acceleration*thermal_expansion*b_factor*
                    (float)std::max(simTempToWorldTemp(p.Temp+5) - simTempToWorldTemp(ambient_T), 0.); // Buoyancy
            u.y += b;
            if(glm::length(u)>2) u = u/glm::length(u)*2.f;
            p.Position += u * timeStep;

            if(p.Life>0.5) p.Life -= m_frame_rate;
            if(timeStep>0) p.Temp = alpha_temp*p.Temp + beta_temp*ambient_T;

        }
    }
}



void Smoke::RespawnParticle(int index, Particle& fire_particle)
{
    float y1 = m_particles[2*index].Position.y;
    float y2 = m_particles[2*index+1].Position.y;
    int offset=0;
    if(m_particles[2*index].Life==0) offset=0;
    else if(m_particles[2*index+1].Life==0) offset=1;
    else if(y1 < y2) offset = 1;

    Particle &particle = m_particles[2*index+offset];
    particle.Life = m_life_max;
    particle.Temp = fire_particle.Temp;
    particle.Position = fire_particle.Position;
    particle.Position.y += 0.8;
    particle.Color = glm::vec4(0.5, 0.5, 0.5, 1.0f);
}

void Smoke::drawParticles(CS123::GL::CS123Shader* shader, OpenGLShape* shape) {
    int i=0;
    glDepthMask(GL_FALSE);
    for (Particle particle : m_particles)
    {
        if (particle.Life > 0.0f)
        {
            glm::mat4 M_fire = glm::translate(glm::mat4(), particle.Position);
            shader->setUniform("m", M_fire);
            shader->setUniform("life", particle.Life);
            vec2 off = m_offset[i];
            shader->setUniform("base_x", off.x);
            shader->setUniform("base_y", off.y);
            shape->drawVAO();
        }
        i++;
    }
    glDepthMask(GL_TRUE);

}
