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
     voxelGrid(VoxelGridDim(vec3(0, 5, 0), 13), 48),
     fireManager(&voxelGrid),
     cloudManager(&voxelGrid)
{
    voxelGrid.getVisualization()->toggle(false, true);

    simulator.init();
    constructShaders();
}

BasicFireScene::~BasicFireScene()
{
}

void BasicFireScene::constructShaders() {

}

std::vector<std::unique_ptr<CS123Shader>> *BasicFireScene::getShaderPrograms(){
    return &shader_bank;
}

void BasicFireScene::render(SupportCanvas3D *context) {
    Voxel* v = voxelGrid.getVoxel(8, 20, 16);
    v->getLastFrameState()->temperature = 20;
    v->getLastFrameState()->q_v = 0.2;
    v = voxelGrid.getVoxel(8, 20, 15);
    v->getLastFrameState()->temperature = 20;
    v->getLastFrameState()->q_v = 0.2;
    v = voxelGrid.getVoxel(9, 20, 16);
    v->getLastFrameState()->temperature = 20;
    v->getLastFrameState()->q_v = 0.2;
    v = voxelGrid.getVoxel(9, 20, 15);
    v->getLastFrameState()->temperature = 20;
    v->getLastFrameState()->q_v = 0.2;
    simulator.step(&voxelGrid);

    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    Camera *camera = context->getCamera();
    voxelGrid.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    voxelGrid.getVisualization()->draw(context);

    fireManager.setCamera(camera->getProjectionMatrix(), camera->getViewMatrix());
    fireManager.setScale(0.03, 0.05);
    fireManager.drawFires(10, settings.boolRenderSmoke);

    cloudManager.setCamera(camera->getProjectionMatrix(), camera->getViewMatrix());
    cloudManager.setScale(0.5);
    cloudManager.draw();

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
    voxelGrid.getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
    voxelGrid.getVisualization()->updateValuesFromSettings();
}


