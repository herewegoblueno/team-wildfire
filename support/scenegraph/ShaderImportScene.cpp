#include "ShaderEvolutionTestingScene.h"
#include "ShaderImportScene.h"
#include "GL/glew.h"
#include <QGLWidget>
#include "glm/gtx/transform.hpp"
#include "support/camera/Camera.h"

#include "support/Settings.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"

#include "support/shapes/Cube.h"
#include "support/shapes/Cone.h"
#include "support/shapes/Sphere.h"
#include "support/shapes/Cylinder.h"
#include "support/shapes/Torus.h"

using namespace CS123::GL;

#include "shaderevolution/ShaderConstructor.h"

ShaderImportScene::ShaderImportScene()
{
    currentShapeIndex = 0;
    currentString = "";
    shapeBank.resize(6);
    defineShapeBank();
    constructShader();
}

ShaderImportScene::~ShaderImportScene()
{
}


void ShaderImportScene::constructShader() {
    if (currentString == "") currentString = "vec3(pos.x,pos.y,pos.z)";
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/shaders/shaderevolutionshader.vert");
    std::string fragmentSource = ShaderConstructor::genShader(currentString);
    current_shader =  std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}


void ShaderImportScene::render(SupportCanvas3D *context) {
    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    current_shader->bind();
    setShaderSceneUniforms(context);
    setLights();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    current_shader->setUniform("m", glm::scale(glm::vec3(2)));
    //Tori are too large...
    if (currentShapeIndex == 3) current_shader->setUniform("m", glm::scale(glm::vec3(1)));

    current_shader->setUniform("time", ShaderEvolutionTestingScene::calculateTime());

    //We don't really need a material actually, evolved shaders don't use lighting
    //current_shader->applyMaterial(mat);

    shapeBank[currentShapeIndex].get()->draw();
    glBindTexture(GL_TEXTURE_2D, 0);
    current_shader->unbind();

    //Trigger another render
    context->update();
}

void ShaderImportScene::setShaderSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    current_shader->setUniform("useLighting", settings.useLighting);
    current_shader->setUniform("p" , camera->getProjectionMatrix());
    current_shader->setUniform("v", camera->getViewMatrix());
}

void ShaderImportScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        current_shader->setLight(lightingInformation[i]);
    }
}

void ShaderImportScene::settingsChanged() {

}


void ShaderImportScene::defineShapeBank(){
    //Can be linked to settings.parameter1-3, but since
    //we know the scenes that are being made we'll but hardcode it
    //Helpful for perfomance reasons too (prevents excessive tessellation)
    int p1 = std::floor(15);
    int p2 = std::floor(15);
    int p3 = std::floor(8);
    shapeBank[0] = std::make_unique<Cube>(p1);
    shapeBank[1] = std::make_unique<Cone>(p1, p2);
    shapeBank[2] = std::make_unique<Cylinder>(p1, p2);
    shapeBank[3] = std::make_unique<Torus>(p1, p2, p3);
    shapeBank[4] = std::make_unique<Sphere>(p1, p2);
    shapeBank[5] = std::make_unique<Sphere>(p1, p2); //Spheres will substitite for meshes
}
