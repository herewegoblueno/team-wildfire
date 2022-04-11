#include "BasicForestScene.h"
#include "GL/glew.h"
#include <QGLWidget>
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

#include <chrono>
using namespace std::chrono;
using namespace CS123::GL;

#include <iostream>

BasicForestScene::BasicForestScene()
{
    defineShapeOptions();
}

BasicForestScene::~BasicForestScene()
{
}

void BasicForestScene::onNewSceneLoaded(){
   constructShaders();
}

void BasicForestScene::constructShaders() {
    shader_bank.clear();
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");

    //THis is here so that with a little bit of modification, you could have different shaders for each primitive
    //TODO: this might need to be improved since if we have a lot of primitives, we'll have a shader for each one of them
    for (int i = 0; i < (int)primitives.size(); i ++){
        std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
        shader_bank.push_back(std::make_unique<CS123Shader>(vertexSource, fragmentSource));
    }
}


std::vector<std::unique_ptr<CS123Shader>> *BasicForestScene::getShaderPrograms(){
    return &shader_bank;
}

void BasicForestScene::render(SupportCanvas3D *context) {
    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int size = primitives.size();
    for (int i = 0; i < size; i++){
        CS123ScenePrimitiveBundle bundle = primitives[i];
        CS123SceneMaterial mat = bundle.primitive.material;
        mat.cDiffuse *= globalData.kd;
        mat.cAmbient *= globalData.ka;
        mat.shininess *= globalData.ks;
        mat.cTransparent *= globalData.kt;

        drawPrimitiveWithShader(i, bundle.model, mat, (shapeOptions[(int) bundle.primitive.type]).get(), context);
    }

    //Trigger another render
    context->update();
}

void BasicForestScene::drawPrimitiveWithShader (int shapeIndex, glm::mat4x4 modelMat, CS123SceneMaterial mat, Shape *shape, SupportCanvas3D *c){
    current_shader = shader_bank[shapeIndex].get();

    current_shader->bind();
    setShaderSceneUniforms(c);
    setLights();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    current_shader->setUniform("m", modelMat);
    current_shader->applyMaterial(mat);
    shape->draw();
    glBindTexture(GL_TEXTURE_2D, 0);
    current_shader->unbind();
}

void BasicForestScene::setShaderSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    current_shader->setUniform("useLighting", settings.useLighting);
    current_shader->setUniform("p" , camera->getProjectionMatrix());
    current_shader->setUniform("v", camera->getViewMatrix());
}

void BasicForestScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        current_shader->setLight(lightingInformation[i]);
    }
}

void BasicForestScene::settingsChanged() {

}


void BasicForestScene::defineShapeOptions(){
    //Can be linked to settings.parameter1-3, but since
    //we know the scenes that are being made we'll but hardcode it
    //Helpful for perfomance reasons too (prevents excessive tessellation)
    int p1 = std::floor(2);
    int p2 = std::floor(2);
    int p3 = std::floor(2);

    shapeOptions.resize(6);
    shapeOptions[0] = std::make_unique<Cube>(p1);
    shapeOptions[1] = std::make_unique<Cone>(p1, p2);
    shapeOptions[2] = std::make_unique<Cylinder>(p1, p2);
    shapeOptions[3] = std::make_unique<Torus>(p1, p2, p3);
    shapeOptions[4] = std::make_unique<Sphere>(p1, p2);
    shapeOptions[5] = std::make_unique<Sphere>(p1, p2); //Spheres will substitite for meshes
}
