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
     voxelGrid(10, vec3(0,5,0), 64),
     fireManager(&voxelGrid)
{

    Voxel* v = voxelGrid.getVoxel(16, 2, 32);
    fireManager.addFire(nullptr, vec3(v->centerInWorldSpace), 30);

    v = voxelGrid.getVoxel(48, 2, 32);
    fireManager.addFire(nullptr, vec3(v->centerInWorldSpace), 30);


    voxelGrid.getVisualization()->toggle(true, true);

    simulator.init();
    constructShaders();
}

BasicFireScene::~BasicFireScene()
{
}

void BasicFireScene::constructShaders() {
    shader_bank.clear();
}

std::vector<std::unique_ptr<CS123Shader>> *BasicFireScene::getShaderPrograms(){
    return &shader_bank;
}

void BasicFireScene::render(SupportCanvas3D *context) {
    voxelGrid.getVisualization()->updateValuesFromSettings();
    Voxel* v = voxelGrid.getVoxel(5, 5, 5);
    v->getLastFrameState()->temperature = 100;
    v->getCurrentState()->temperature = 100;
    simulator.step(&voxelGrid);

    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    Camera *camera = context->getCamera();
    voxelGrid.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    voxelGrid.getVisualization()->draw(context);

    fireManager.setCamera(camera->getProjectionMatrix(), camera->getViewMatrix());
    fireManager.setScale(0.03, 0.05);
    fireManager.drawFires(10, false);

    //Trigger another render
    simulator.cleanupForNextStep(&voxelGrid);
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


