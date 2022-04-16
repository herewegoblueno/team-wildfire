#include "BasicFireScene.h"
#include "GL/glew.h"
#include <QGLWidget>
#include "support/camera/Camera.h"

#include "support/Settings.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"

#include "support/shapes/Trunk.h"
#include "support/shapes/Leaf.h"

#include <chrono>
using namespace std::chrono;
using namespace CS123::GL;

#include <iostream>

BasicFireScene::BasicFireScene():
     voxelGrids(3, vec3(0,0,0), 30)
{
    fire = std::make_unique<Fire> (10, glm::vec3(0));
    voxelGrids.getVisualization()->toggle(false);
    constructShaders();
}

BasicFireScene::~BasicFireScene()
{
}

void BasicFireScene::onNewSceneLoaded(){
   constructShaders();

}

void BasicFireScene::constructShaders() {
    shader_bank.clear();
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");

    //This is here so that with a little bit of modification, you could have different shaders for each primitive
    //TODO: this might need to be improved since if we have a lot of primitives, we'll have a shader for each one of them
    for (int i = 0; i < 1; i ++){
        std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
        shader_bank.push_back(std::make_unique<CS123Shader>(vertexSource, fragmentSource));
    }
}


std::vector<std::unique_ptr<CS123Shader>> *BasicFireScene::getShaderPrograms(){
    return &shader_bank;
}

void BasicFireScene::render(SupportCanvas3D *context) {
    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    std::cout << "render fire" << std::endl<< std::flush;

    current_shader = shader_bank[0].get();
//    voxelGrids.getVisualization()->toggle(true);

    current_shader->bind();
    Camera *camera = context->getCamera();
    voxelGrids.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    voxelGrids.getVisualization()->draw(context);
    current_shader->unbind();


    fire->setViewProjection(camera->getViewMatrix(), camera->getProjectionMatrix());
    fire->drawParticles();

    //Trigger another render
    context->update();
}

void BasicFireScene::drawPrimitiveWithShader(int shapeIndex, glm::mat4x4 modelMat, CS123SceneMaterial mat, SupportCanvas3D *c)
{
    current_shader = shader_bank[shapeIndex].get();

    //TODO: will have to minimize bindings and unbindings to optimize
    //We'd want to draw all the things that use any given shader all at once
    current_shader->bind();
    setShaderSceneUniforms(c);
    setLights();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    current_shader->setUniform("m", modelMat);
    current_shader->applyMaterial(mat);

    glBindTexture(GL_TEXTURE_2D, 0);
    current_shader->unbind();
}

void BasicFireScene::setShaderSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    current_shader->setUniform("useLighting", settings.useLighting);
    current_shader->setUniform("p" , camera->getProjectionMatrix());
    current_shader->setUniform("v", camera->getViewMatrix());
}

void BasicFireScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        current_shader->setLight(lightingInformation[i]);
    }
}

void BasicFireScene::settingsChanged() {

}


