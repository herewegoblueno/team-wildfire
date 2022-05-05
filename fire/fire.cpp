#include "support/shapes/OpenGLShape.h"
#include "support/gl/textures/Texture2D.h"
#include "support/gl/textures/TextureParametersBuilder.h"

#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/FBO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"
#include "support/gl/shaders/CS123Shader.h"


#include "support/lib/ResourceLoader.h"
#include <glm/gtx/transform.hpp>

#include "fire/fire.h"
#include "voxels/voxelgrid.h"
#include "simulation/physics.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;


Fire::Fire(int density, glm::vec3 center, float size, VoxelGrid* grid):
    m_density(density), m_size(size), m_center(center), m_grid(grid)
{
    // init particles and preset speed
    m_particles.clear();
    m_vels.clear();
    m_poss.clear();
    dist = std::normal_distribution<float>(mean, stddev);
    for (unsigned int i = 0; i < density; ++i)
    {
        float random_x = dist(generator) / 10.0f;
        float random_y = dist(generator) / 40.0f;
        float random_z = dist(generator) / 10.0f;
        float vec_y = (rand() % 100 - 50)/ 150.0f;
        float vec_x = (rand() % 100 - 50)/ 150.0f * (1-vec_y);
        float vec_z = (rand() % 100 - 50)/ 150.0f * (1-vec_y);
        m_particles.push_back(Particle());

        m_poss.push_back(glm::vec3(random_x, random_y, random_z));
        m_vels.push_back(glm::vec3(vec_x, vec_y, vec_z));
    }
    //set respawn rate based on given density
    assert(m_density>2);
    m_respawn_num = fire_frame_rate * m_density / m_life;
    m_respawn_num = std::max(m_respawn_num, 2);

    // init renderer
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // init smoke
    m_smoke = std::make_unique<Smoke>(density, fire_frame_rate, size, m_grid);
}


Fire::~Fire()
{
}

void Fire::setSize(float size) {
    m_size = size;
}

void Fire::update_particles()
{
    unsigned int nr_new_particles = m_respawn_num;
    // add new particles
    for (unsigned int i = 0; i < nr_new_particles; ++i)
    {
        int unusedParticle = FirstUnusedParticle();
        if(unusedParticle<m_density)
            RespawnParticle(unusedParticle);
    }
    // update all particles
    VoxelPhysicalData voxel;
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
//        p.Life -= fire_frame_rate*p.Temp*burn_coef;
        #ifdef CUDA_FLUID
        p.Life -= fire_frame_rate*0.01;
        #else
        p.Life -= fire_frame_rate;
        #endif

        if (p.Life > 0.0f)
        {
            // particle is alive, thus update
            voxel = m_grid->getStateInterpolatePoint(p.Position);
            VoxelPhysicalData* vox = &voxel;
            float ambient_T = vox->temperature;
            if(std::isnan(ambient_T)) ambient_T = 0;

            float b_factor = 0.15;

            glm::vec3 u = vec3(vox->u);
            #ifdef CUDA_FLUID
            b_factor = 0.0;
            #else
            p.Temp = alpha_temp*p.Temp + beta_temp*ambient_T;
            #endif
            glm::vec3 b = glm::vec3(0, gravity_acceleration*thermal_expansion*b_factor, 0)*
                    (float)(simTempToWorldTemp(p.Temp) - simTempToWorldTemp(ambient_T)); // Buoyancy

            p.Position += (b + u) * fire_frame_rate*0.5f;

            if(p.Life < fire_frame_rate*1.5 || p.Temp < 10)
            {
//                m_smoke->RespawnParticle(i, p);
                p.Life = 0;
            }
        }
    }
}



void Fire::RespawnParticle(int index)
{
    Particle &particle = m_particles[index];
    glm::vec3 offset = m_poss[index];
    offset = offset*5.f*m_size;

    float rColor = 0.5f + ((rand() % 50) / 100.0f);
    particle.Position = m_center + offset;
    particle.Color = glm::vec4(rColor, rColor, rColor, 1.0f);
    particle.Life = m_life;
    particle.Temp = 15;

    particle.Velocity = m_size*m_vels[index];
}



void Fire::drawSmoke(CS123::GL::CS123Shader* shader, OpenGLShape* shape) {
    m_smoke->drawParticles(shader, shape);
}

void Fire::drawParticles(CS123::GL::CS123Shader* shader, OpenGLShape* shape) {
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
