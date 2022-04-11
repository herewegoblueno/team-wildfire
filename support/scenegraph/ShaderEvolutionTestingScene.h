#ifndef SHADEREVOLUTIONTESTINGSCENE_H
#define SHADEREVOLUTIONTESTINGSCENE_H


#include "OpenGLScene.h"

#include <memory>
#include <vector>
#include "support/shapes/Shape.h"
#include "shaderevolution/AstNodes.h"


namespace CS123 { namespace GL {

    class Shader;
    class CS123Shader;
    class Texture2D;
}}


class ShaderEvolutionTestingScene : public OpenGLScene {
public:
    ShaderEvolutionTestingScene();
    virtual ~ShaderEvolutionTestingScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

    void initializeGenotypes();
    void constructShaders();

    static int numberOfTestShaders;
    static float calculateTime();
    static long startTime;

    std::string getShaderSource(int index, bool showAnnotations);
    std::vector<std::unique_ptr<ShaderGenotype>>* getShaderGenotypes();
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> * getShaderPrograms();

private:

    void setShaderSceneUniforms(SupportCanvas3D *context);
    void setLights();
    void defineShapeBank();
    void drawPrimitiveWithShader(int shapeIndex, glm::mat4x4 modelMat, CS123SceneMaterial mat, Shape *shape, SupportCanvas3D *c);

    std::vector<std::unique_ptr<Shape>> shapeBank;
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> shader_bank;
    std::vector<std::unique_ptr<ShaderGenotype>> genotype_bank;
    CS123::GL::CS123Shader *current_shader;

};


#endif // SHADEREVOLUTIONTESTINGSCENE_H
