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
        float random_y = 0;
        float random_z = dist(generator) / 10.0f;
        float vec_y = (rand() % 50)/ 50.0f;
        float vec_x = (rand() % 100 - 50)/ 150.0f * (1-vec_y);
        float vec_z = (rand() % 100 - 50)/ 150.0f * (1-vec_y);
        m_particles.push_back(Particle());
        m_poss.push_back(glm::vec3(random_x, random_y, random_z));
        m_vels.push_back(glm::vec3(vec_x, vec_y*0.3, vec_z));
    }
    //set respawn rate based on given density
    assert(m_density>2);
    m_respawn_num = fire_frame_rate * m_density / m_life;

    // init renderer
    InitRender();
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // init smoke
    m_smoke = std::make_unique<Smoke>(density, fire_frame_rate, size, m_grid);
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
    m_texture = std::make_unique<Texture2D>(gl_img.bits(), gl_img.width(), gl_img.height());
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
            RespawnParticle(unusedParticle);
    }
    // update all particles
    for (unsigned int i = 0; i < m_density; ++i)
    {
        Particle &p = m_particles[i];
//        p.Life -= fire_frame_rate*p.Temp*burn_coef;
        p.Life -= fire_frame_rate;

        if (p.Life > 0.0f)
        {
            // particle is alive, thus update
            Voxel* voxel = m_grid->getVoxelClosestToPoint(p.Position);
            VoxelPhysicalData* vox = voxel->getCurrentState();
            float ambient_T = vox->temperature;

            if(std::isnan(ambient_T)) ambient_T = 0;

            float x = p.Position.x;
            float y = p.Position.y;
            float z = p.Position.z;
            float te = p.Temp;

            glm::vec3 u = vec3(vox->u);
            float c_dis = glm::distance(p.Position, m_center)+0.001f;
            u = glm::normalize(p.Position + glm::vec3(0,1,0) - m_center)*std::min(0.05f+0.2f/c_dis, 0.1f);

            glm::vec3 b = -glm::vec3(0, gravity_acceleration*thermal_expansion, 0)*(p.Temp - ambient_T); // Buoyancy

            p.Position += (b+u) * fire_frame_rate;

            if(std::isnan(p.Position.x))
            {
                cout << x << " " << y << " " << z << " " << te<< endl << flush;
                cout << (b+u).x << " " << (b+u).y << " " << (b+u).z << endl << flush;
                cout << "crash loop";
            }

            glm::vec3 adjust_vec = p.Position - m_center;
            adjust_vec.y = adjust_vec.y*0.5;
            float adjust_len = glm::length(adjust_vec);
            float neighbor_temp = 5 + 10*std::exp(-0.5*adjust_len*adjust_len/0.005);
            p.Temp = alpha_temp*p.Temp + beta_temp*(neighbor_temp + ambient_T);

            if(p.Life < fire_frame_rate*1.5 || p.Temp < 10)
            {
                m_smoke->RespawnParticle(i, p);
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



void Fire::drawSmoke(CS123::GL::CS123Shader* shader) {
    m_smoke->drawParticles(shader);
}

void Fire::drawParticles(CS123::GL::CS123Shader* shader) {
    update_particles();

    // bind texture (though currently it's not even used in the shader since it doesn't work on Mac)
    TextureParametersBuilder builder;
    builder.setFilter(TextureParameters::FILTER_METHOD::LINEAR);
    builder.setWrap(TextureParameters::WRAP_METHOD::REPEAT);

    TextureParameters parameters = builder.build();
    parameters.applyTo(*m_texture);
    std::string filename = "sprite";
    shader->setTexture(filename, *m_texture);

    for (Particle particle : m_particles)
    {
        if (particle.Life > 0.0f)
        {
            shader->setUniform("color", particle.Color);
            glm::mat4 M_fire = glm::translate(glm::mat4(), particle.Position);
            shader->setUniform("m", M_fire);
            shader->setUniform("temp", particle.Temp);

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
