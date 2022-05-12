#ifndef CloudManager_H
#define CloudManager_H

#include <map>
#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>

#include "voxels/voxelgrid.h"
#include "particle.h"
#include "fire.h"
#include "smoke.h"
#include "trees/module.h"



class CloudManager
{
public:
    CloudManager(VoxelGrid *grid);
    void draw();

    void setScale(float s);
    void setCamera(glm::mat4 projection, glm::mat4 view);

private:
    VoxelGrid *m_grid;
    std::multimap<Module *, std::shared_ptr<Fire>> m_fires;

    const float mean = 0.0;
    const float stddev = 0.2;
    std::default_random_engine generator;
    std::normal_distribution<float> dist;
    // render components
    std::unique_ptr<OpenGLShape> m_quad;

    glm::mat4 p;
    glm::mat4 v;
    float scale = 0.3;

    std::unique_ptr<CS123::GL::CS123Shader> m_shader;
};

#endif // CloudManager_H
