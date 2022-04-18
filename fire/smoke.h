#ifndef SMOKE_H
#define SMOKE_H

#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>

#include "particle.h"

namespace CS123 { namespace GL {
    class Shader;
    class CS123Shader;
    class Texture2D;
}}

class OpenGLShape;
class VoxelGrid;


class Smoke
{
public:
    Smoke(int density, float frame_rate, float size, VoxelGrid* grid);
    ~Smoke();

    void drawParticles(CS123::GL::CS123Shader* shader);
    void update_particles();
    void RespawnParticle(int index, Particle& fire_particle);

private:
    int m_density;
    float m_size;
    float m_frame_rate;
    float m_life_max = 5.0f;
    VoxelGrid* m_grid;
    std::vector<Particle> m_particles;

    glm::vec3 m_center;

    std::unique_ptr<CS123::GL::Texture2D> m_texture;

    void InitRender();

    std::unique_ptr<OpenGLShape> m_quad;

};





#endif
