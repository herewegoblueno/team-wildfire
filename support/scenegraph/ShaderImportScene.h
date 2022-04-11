#ifndef SHADERIMPORTSCENE_H
#define SHADERIMPORTSCENE_H


#include "OpenGLScene.h"

#include <vector>
#include "support/shapes/Shape.h"

namespace CS123 { namespace GL {

    class Shader;
    class CS123Shader;
    class Texture2D;
}}


class ShaderImportScene : public OpenGLScene {
public:
    ShaderImportScene();
    virtual ~ShaderImportScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

    void constructShader();
    std::string currentString;
    int currentShapeIndex;

private:

    void setShaderSceneUniforms(SupportCanvas3D *context);
    void setLights();
    void defineShapeBank();

    std::vector<std::unique_ptr<Shape>> shapeBank;
    std::unique_ptr<CS123::GL::CS123Shader> current_shader;
};


#endif // SHADERIMPORTSCENE_H
