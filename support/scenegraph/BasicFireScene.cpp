#include "GL/glew.h"
#include <QGLWidget>
#include "support/camera/Camera.h"

#include "BasicFireScene.h"
#include "support/Settings.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"

#include "support/shapes/Trunk.h"
#include "support/shapes/Leaf.h"

#include <chrono>
#include <iostream>

using namespace std::chrono;
using namespace CS123::GL;


BasicFireScene::BasicFireScene():
     voxelGrids(8, vec3(0,0,0), 52)
{
    fires.clear();
    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(-0.8, -0.5, 1.2), 0.6, &voxelGrids) );

//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(0, -1, 0), 0.5, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(0, -1, 0), 0.6, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(0, -1, 2), 0.5, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(0, -1, 2), 0.6, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(-0.8, -0.5, 1.2), 0.5, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(1.6, -0.3, -1.2), 0.5, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(1.6, -0.3, -1.2), 0.6, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(2, -0.5, 0.8), 0.5, &voxelGrids) );
//    fires.push_back(  std::make_unique<Fire> (500, glm::vec3(2, -0.5, 0.8), 0.6, &voxelGrids) );

    voxelGrids.getVisualization()->toggle(false, true);

    simulator.init();
    constructShaders();
}

BasicFireScene::~BasicFireScene()
{
}

void BasicFireScene::constructShaders() {
    shader_bank.clear();

    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/particle.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/fire.frag");
    shader_bank.push_back(std::make_unique<CS123Shader>(vertexSource, fragmentSource));

    vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/particle.vert");
    fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/smoke.frag");
    shader_bank.push_back(std::make_unique<CS123Shader>(vertexSource, fragmentSource));
}


std::vector<std::unique_ptr<CS123Shader>> *BasicFireScene::getShaderPrograms(){
    return &shader_bank;
}

void BasicFireScene::render(SupportCanvas3D *context) {
    simulator.linear_step(&voxelGrids);

    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    current_shader = shader_bank[0].get();

//    current_shader->bind();
    Camera *camera = context->getCamera();
    voxelGrids.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    voxelGrids.getVisualization()->draw(context);
//    current_shader->unbind();

    std::vector<std::unique_ptr<CS123Shader>> *shaders = getShaderPrograms();
    CS123Shader* fire_shader = shaders->at(0).get();
    CS123Shader* smoke_shader = shaders->at(1).get();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    fire_shader->bind();
    fire_shader->setUniform("p", camera->getProjectionMatrix());
    fire_shader->setUniform("v", camera->getViewMatrix());
    smoke_shader->setUniform("scale", 0.03f);

    int len = fires.size();
    for (int f=0; f<len;f++)
    {
        fires[f]->drawParticles(fire_shader);
    }
    fire_shader->unbind();

    smoke_shader->bind();
    smoke_shader->setUniform("p", camera->getProjectionMatrix());
    smoke_shader->setUniform("v", camera->getViewMatrix());
    smoke_shader->setUniform("scale", 0.05f);

    for (int f=0; f<len;f++)
    {
        fires[f]->drawSmoke(smoke_shader);
    }
    smoke_shader->unbind();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);

    //Trigger another render
    simulator.cleanupForNextStep(&voxelGrids);
    context->update();
}

void BasicFireScene::setShaderSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    current_shader->setUniform("useLighting", settings.useLighting);
    current_shader->setUniform("p" , camera->getProjectionMatrix());
    current_shader->setUniform("v", camera->getViewMatrix());
}


void BasicFireScene::settingsChanged() {

}


