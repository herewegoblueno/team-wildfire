#ifndef BASICFORESTSCENE_H
#define BASICFORESTSCENE_H


#include "OpenGLScene.h"

#include <memory>
#include <vector>
#include "support/shapes/Shape.h"
#include "voxels/voxelgridline.h"


namespace CS123 { namespace GL {

    class Shader;
    class CS123Shader;
    class Texture2D;
}}


class BasicForestScene : public OpenGLScene {
public:
    BasicForestScene();
    virtual ~BasicForestScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;
    void onNewSceneLoaded();

    void constructShaders();
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> * getShaderPrograms();

private:

    void setShaderSceneUniforms(SupportCanvas3D *context);
    void setLights();
    void defineShapeOptions();
    void drawPrimitiveWithShader(int shapeIndex, glm::mat4x4 modelMat, CS123SceneMaterial mat, Shape *shape, SupportCanvas3D *c);

    std::vector<std::unique_ptr<Shape>> shapeOptions;
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> shader_bank;
    CS123::GL::CS123Shader *current_shader;
    VoxelGridLine gridline;

};


#endif // BASICFORESTSCENE_H
