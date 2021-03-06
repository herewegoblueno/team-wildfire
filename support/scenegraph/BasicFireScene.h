#ifndef BASICFIRESCENE_H
#define BASICFIRESCENE_H

#include <memory>
#include <vector>
#include "OpenGLScene.h"
#include "voxels/voxelgrid.h"
#include "fire/firemanager.h"
#include "fire/cloudmanager.h"
#include "simulation/simulator.h"

class BasicFireScene : public OpenGLScene {
public:
    BasicFireScene();
    virtual ~BasicFireScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;
    void constructShaders();
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> * getShaderPrograms();

private:

    void setShaderSceneUniforms(SupportCanvas3D *context);

    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> shader_bank;
    CS123::GL::CS123Shader *current_shader;
    VoxelGrid voxelGrid;
    FireManager fireManager;
    CloudManager cloudManager;

    Simulator simulator;
};


#endif // BASICFIRESCENE_H
