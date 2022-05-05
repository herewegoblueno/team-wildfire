#ifndef FIRE_H
#define FIRE_H

#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>

#include "particle.h"
#include "smoke.h"


class Fire
{
public:
    Fire(int density, glm::vec3 center, float size, VoxelGrid* grid);
    ~Fire();

    void drawParticles( CS123::GL::CS123Shader* shader, OpenGLShape* shape);
    void drawSmoke( CS123::GL::CS123Shader* shader, OpenGLShape* shape);
    void setSize(float size);

private:
    // particle cluster property
    int m_density;
    float m_size;
    float fire_frame_rate = 0.03;
    float m_life = 3.0f;
    int m_respawn_num = 2;
    glm::vec3 m_center;
    std::vector<glm::vec3> m_vels;
    std::vector<glm::vec3> m_poss;
    std::vector<Particle> m_particles;
    std::unique_ptr<Smoke> m_smoke;

    // particle life cycle
    unsigned int lastUsedParticle = 0;
    void update_particles();
    unsigned int FirstUnusedParticle();
    void RespawnParticle(int index);

    // render components
    VoxelGrid* m_grid;

    // gaussian random generator
    const float mean = 0.0;
    const float stddev = 0.2;
    std::default_random_engine generator;
    std::normal_distribution<float> dist;

};





#endif
