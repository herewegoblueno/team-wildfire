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
#include "voxels/voxelgrid.h"
#include "simulation/physics.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

Smoke::Smoke(int density, float frame_rate, float size, VoxelGrid* grid):
    m_density(density*2), m_size(size), m_frame_rate(frame_rate), m_grid(grid)
{
    m_particles.clear();
    for (unsigned int i = 0; i < m_density; ++i)
        m_particles.push_back(Particle());

}



Smoke::~Smoke()
{

}


void Smoke::update_particles()
{
    // update all particles
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
        if (p.Life > 0.9f)
        {	// particle is alive, thus update
            VoxelPhysicalData* vox = m_grid->getVoxelClosestToPoint(p.Position)->getCurrentState();

            glm::vec3 u = vec3(vox->u);
            float ambient_T = vox->temperature;

            float b_factor = 0.5;
            #ifdef CUDA_FLUID
            b_factor = 0.2;
            #else
            p.Temp = alpha_temp*p.Temp + beta_temp*ambient_T;
            #endif
            float b = gravity_acceleration*thermal_expansion*b_factor*
                    (float)std::max(simTempToWorldTemp(p.Temp) - simTempToWorldTemp(ambient_T), 0.); // Buoyancy
            u.y += b;
            p.Position += u * m_frame_rate;

            p.Position += (b+u+p.Velocity) * m_frame_rate;
            p.Temp = alpha_temp*p.Temp + beta_temp*(ambient_T);

            p.Velocity = p.Velocity*0.9f;
        }
    }
}



void Smoke::RespawnParticle(int index, Particle& fire_particle)
{

    float vec_x = (rand() % 100 - 50)/ 120.0f * 0.2;
    float vec_z = (rand() % 100 - 50)/ 120.0f * 0.2;

    glm::vec3 turbulance(vec_x, 0, vec_z);

    Particle &particle = m_particles[2*index];
    particle.Life = 1;
    particle.Temp = fire_particle.Temp;
    particle.Position = fire_particle.Position;
    particle.Velocity = turbulance;
    particle.Color = glm::vec4(0.5, 0.5, 0.5, 1.0f);

    particle = m_particles[2*index+1];
    particle.Life = 1;
    particle.Temp = fire_particle.Temp;
    particle.Position = fire_particle.Position;
    particle.Velocity = - turbulance;
    particle.Color = glm::vec4(0.5, 0.5, 0.5, 1.0f);
}

void Smoke::drawParticles(CS123::GL::CS123Shader* shader, OpenGLShape* shape) {
    update_particles();

    for (Particle particle : m_particles)
    {
        if (particle.Life > 0.0f)
        {
            shader->setUniform("color", particle.Color);
            glm::mat4 M_fire = glm::translate(glm::mat4(), particle.Position);
            shader->setUniform("m", M_fire);
            shader->setUniform("temp", particle.Temp);

            shape->drawVAO();
        }
    }

}
