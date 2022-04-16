#ifndef BASICFIRESCENE_H
#define BASICFIRESCENE_H


#include "OpenGLScene.h"
#include "voxels/voxelgrid.h"

#include <memory>
#include <vector>

#include "fire/fire.h"

namespace CS123 { namespace GL {
    class Shader;
    class CS123Shader;
    class Texture2D;
}}


class BasicFireScene : public OpenGLScene {
public:
    BasicFireScene();
    virtual ~BasicFireScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;
    void onNewSceneLoaded();

    void constructShaders();
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> * getShaderPrograms();

private:
     std::vector<std::unique_ptr<Fire>> fires;

    void setShaderSceneUniforms(SupportCanvas3D *context);
    void setLights();
    void drawPrimitiveWithShader(int shapeIndex, glm::mat4x4 modelMat, CS123SceneMaterial mat, SupportCanvas3D *c);

    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> shader_bank;
    CS123::GL::CS123Shader *current_shader;
    VoxelGrid voxelGrids;

};


#endif // BASICFIRESCENE_H
