#include "BasicForestScene.h"
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

BasicForestScene::BasicForestScene(MainWindow *mainWindow):
     _voxelGrid(forestWidth + gridBuffer, vec3(0,0,0), 60),
     mainWindow(mainWindow)
{
    loadPhongShader();
    initDebugMaterials();
    tessellateShapes();
    _voxelGrid.getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
    _forest = std::make_unique<Forest>(&_voxelGrid, numTrees, forestWidth, forestHeight);
    //The forest also initializes the mass of the voxels
    updatePrimitivesFromForest();
    _simulator.init();
    _voxelGrid.getVisualization()->setForestReference(_forest.get());
    mainWindow->updateModuleSelectionOptions(_forest->getAllModuleIDs());
}

BasicForestScene::~BasicForestScene()
{
}

void BasicForestScene::updatePrimitivesFromForest() {
    _leaves.clear();
    _trunks.clear();
    _forest->recalculatePrimitives();
    std::vector<PrimitiveBundle> forestPrimitives = _forest->getPrimitives();
    PrimitiveType type;
    for (PrimitiveBundle &bundle : forestPrimitives) {
        type = bundle.primitive.type;
        if (type == PrimitiveType::PRIMITIVE_TRUNK) {
            _trunks.push_back(bundle);
        } else if (type == PrimitiveType::PRIMITIVE_LEAF) {
            _leaves.push_back(bundle);
        }
    }
}

void BasicForestScene::loadPhongShader() {
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
    _phongShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}

void BasicForestScene::render(SupportCanvas3D *context) {

    _simulator.step(&_voxelGrid, _forest.get());

    Camera *camera = context->getCamera();
    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _phongShader->bind();
    setGlobalData();
    setSceneUniforms(context);
    setLights();
    renderGeometry();
    glBindTexture(GL_TEXTURE_2D, 0);
    _phongShader->unbind();

    _voxelGrid.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    _voxelGrid.getVisualization()->draw(context);

    _simulator.cleanupForNextStep(&_voxelGrid, _forest.get());
    updatePrimitivesFromForest();
    //Trigger another render
    context->update();
}

/**
 *  Tessellate scene shapes and store them for later rendering
 */
void BasicForestScene::tessellateShapes() {
    _leaf = std::make_unique<Leaf>();
    _trunk = std::make_unique<Trunk>(1, 1);
}

/** Render each primitive's shape */
void BasicForestScene::renderGeometry() {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    if (settings.seeBranchModules) {
        renderTrunksVisualizedModules();
    } else {
        renderTrunks();
        renderLeaves();
    }
}

void BasicForestScene::initDebugMaterials() {
    // Neon green
    _selectedBranchMat.cAmbient.r = 0.22;
    _selectedBranchMat.cAmbient.g = 1;
    _selectedBranchMat.cAmbient.b = 0.1;
    // Bright red
    _warningBranchesMat.cAmbient.r = 1.0;
    _warningBranchesMat.cAmbient.g = 0.1;
    _warningBranchesMat.cAmbient.b = 0.1;
}

/** Color each branch based on the module it's in */
void BasicForestScene::renderTrunksVisualizedModules() {
    _trunk->bindVAO();
    for (PrimitiveBundle &bundle : _trunks) {
        _phongShader->setUniform("m",  bundle.model);
        int moduleID = bundle.moduleID;
        CS123SceneMaterial mat;
        if (_moduleIDToMat.count(moduleID)) {
            //We've already made a material for this module
            mat = _moduleIDToMat[moduleID];
        } else {
            //We have to first make a material for this module
            mat = bundle.primitive.material;
            mat.cAmbient.r = randomDarkColor();
            mat.cAmbient.g = randomDarkColor() / 2; //Since selected modules will be neon green, want to differentiate more
            mat.cAmbient.b = randomDarkColor();
            _moduleIDToMat[moduleID] = mat;
        }
        if (moduleID == settings.selectedModuleId) {
            mat = _selectedBranchMat;
        } else if (bundle.warning) {
            mat = _warningBranchesMat;
        }
        _phongShader->applyMaterial(mat);
        _trunk->drawVAO();
    }
    _trunk->unbindVAO();
}

void BasicForestScene::renderTrunks() {
    _trunk->bindVAO();
    for (PrimitiveBundle &bundle : _trunks) {
        _phongShader->setUniform("m",  bundle.model);
        _phongShader->applyMaterial(bundle.primitive.material);
        _trunk->drawVAO();
    }
    _trunk->unbindVAO();
}

void BasicForestScene::renderLeaves() {
    _leaf->bindVAO();
    for (PrimitiveBundle &bundle : _leaves) {
        _phongShader->setUniform("m",  bundle.model);
        _phongShader->applyMaterial(bundle.primitive.material);
        _leaf->drawVAO();
    }
    _leaf->unbindVAO();
}

void BasicForestScene::defineLights() {
    lightingInformation.clear();
    CS123SceneLightData light;
    light.type = LightType::LIGHT_DIRECTIONAL;
    light.dir = glm::normalize(glm::vec4(1.f, -1.f, -1.f, 0.f));
    light.color.r = light.color.g = light.color.b = 1;
    light.id = 0;
    lightingInformation.push_back(light);
}

void BasicForestScene::defineGlobalData() {
    globalData.ka = 1.0f;
    globalData.kd = 1.0f;
    globalData.ks = 1.0f;
}

void BasicForestScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        _phongShader->setLight(lightingInformation[i]);
    }
}

void BasicForestScene::setGlobalData(){
    _phongShader->setUniform("ka", globalData.ka);
    _phongShader->setUniform("kd", globalData.kd);
    _phongShader->setUniform("ks", globalData.ks);
}

void BasicForestScene::setSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    _phongShader->setUniform("useLighting", settings.useLighting);
    _phongShader->setUniform("p" , camera->getProjectionMatrix());
    _phongShader->setUniform("v", camera->getViewMatrix());
}

void BasicForestScene::settingsChanged() {
     _voxelGrid.getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
     _voxelGrid.getVisualization()->updateValuesFromSettings();
}
