#ifndef FIREMANAGER_H
#define FIREMANAGER_H

#include <map>
#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>

#include "support/gl/textures/Texture2D.h"
#include "voxels/voxelgrid.h"
#include "particle.h"
#include "fire.h"
#include "smoke.h"
#include "trees/module.h"

inline int densityFromSize(float size) {
#ifdef CUDA_FLUID
    return 200;
#endif
    return size*size*1000;
}

class FireManager
{
public:
    FireManager(VoxelGrid *grid);
    void addFire(Module *m, glm::vec3 pos, float size);
    double massChangeRateToFireSize(double massChangeRate);
    void setModuleFireSizes(Module *m, float size);
    void removeFires(Module *m);
    void drawFires(float time, bool smoke=true);

    void setScale(float fire_particle_size, float smoke_particle_size);
    void setCamera(glm::mat4 projection, glm::mat4 view);

private:
    VoxelGrid *m_grid;
    std::multimap<Module *, std::shared_ptr<Fire>> m_fires;

    // render components
    std::unique_ptr<OpenGLShape> m_quad;
    std::unique_ptr<CS123::GL::Texture2D> m_texture;

    glm::mat4 p;
    glm::mat4 v;
    float scale_fire = 0.3;
    float scale_smoke = 0.5;

    std::unique_ptr<CS123::GL::CS123Shader> m_fireshader;
    std::unique_ptr<CS123::GL::CS123Shader> m_smokeshader;
};

#endif // FIREMANAGER_H
