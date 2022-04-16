#ifndef FIRE_H
#define FIRE_H

#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>

namespace CS123 { namespace GL {
    class Shader;
    class CS123Shader;
    class Texture2D;
}}

class OpenGLShape;

struct Particle {
    glm::vec3 Position, Velocity;
    glm::vec4 Color;
    float     Life;

    Particle()
      : Position(0.0f), Velocity(0.0f), Color(1.0f), Life(0.0f) { }
};


class Fire
{
public:
    Fire(int density, glm::vec3 center);
    ~Fire();

    void setViewProjection(glm::mat4x4 v, glm::mat4x4 p);
    void drawParticles();

private:
    int m_density;
    float fire_frame_rate = 0.1;
    float m_life = 5.0f;
    int m_respawn_num = 2;
    std::vector<Particle> m_particles;

    glm::mat4x4 m_p;
    glm::mat4x4 m_v;
    glm::vec3 m_center;
    unsigned int lastUsedParticle = 0;
    void update_particles();
    unsigned int FirstUnusedParticle();
    void RespawnParticle(Particle &particle);

    std::unique_ptr<CS123::GL::Texture2D> m_texture;

    void InitRender();

    std::unique_ptr<CS123::GL::CS123Shader> m_particleUpdateProgram;
    std::unique_ptr<CS123::GL::CS123Shader> m_particleDrawProgram;

    std::unique_ptr<OpenGLShape> m_quad;
    GLuint m_particlesVAO;


    const float mean = 0.0;
    const float stddev = 0.3;
    std::default_random_engine generator;
    std::normal_distribution<float> dist;

};





#endif
