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
#include <iostream>
using namespace std::chrono;
using namespace CS123::GL;



BasicForestScene::BasicForestScene(MainWindow *mainWindow):
     _voxelGrid(forestWidth + gridBuffer, vec3(0,0,0), 40),
     mainWindow(mainWindow)
{
    loadShaders();
    tessellateShapes();
    _voxelGrid.getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
    _forest = std::make_unique<Forest>(&_voxelGrid, numTrees, forestWidth, forestHeight);
    _lastFrameNumModules = _forest->getAllModuleIDs().size();
    //The forest also initializes the mass of the voxels
    updatePrimitivesFromForest();
    _simulator.init();
    _voxelGrid.getVisualization()->setForestReference(_forest.get());
    mainWindow->updateModuleSelectionOptions(_forest->getAllModuleIDs());

    vec3 fire_center(0,0,0);
    Voxel* v = _voxelGrid.getVoxelClosestToPoint(fire_center);
    _fire_mngr.addFire(v, fire_center, 0.8);

    fire_center = vec3 (1,1,1);
    v = _voxelGrid.getVoxelClosestToPoint(fire_center);
    _fire_mngr.addFire(v, fire_center, 0.8);

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

void BasicForestScene::loadShaders() {
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
    _phongShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);

    fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/module.frag");
    _moduleVisShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}

void BasicForestScene::render(SupportCanvas3D *context) {
    _simulator.step(&_voxelGrid, _forest.get());

    Camera *camera = context->getCamera();
    glClearColor(0.2, 0.2, 0.2, 0.3);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    CS123Shader *selectedShader = settings.seeBranchModules ? _moduleVisShader.get() : _phongShader.get();

    selectedShader->bind();
    setGlobalData(selectedShader);
    setSceneUniforms(context, selectedShader);
    setLights(selectedShader);
    renderGeometry();
    glBindTexture(GL_TEXTURE_2D, 0);
    selectedShader->unbind();

    _voxelGrid.getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    _voxelGrid.getVisualization()->draw(context);

    _fire_mngr.setCamera(camera->getProjectionMatrix(), camera->getViewMatrix());
    _fire_mngr.setScale(0.03, 0.05);
    _fire_mngr.drawFire(false);
    // Clean up dead modules, update radii, update UI
    _simulator.cleanupForNextStep(&_voxelGrid, _forest.get());
    updatePrimitivesFromForest();
    std::vector<int> moduleIDs = _forest->getAllModuleIDs();
    uint numModules = moduleIDs.size();
    if (numModules < _lastFrameNumModules) {
        mainWindow->updateModuleSelectionOptions(moduleIDs);
    }
    _lastFrameNumModules = numModules;
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

/** Color each branch based on the module it's in */
void BasicForestScene::renderTrunksVisualizedModules() {
    _trunk->bindVAO();
    for (PrimitiveBundle &bundle : _trunks) {
        int moduleID = bundle.moduleID;
        Module *module = _forest->getModuleFromId(moduleID);

        _moduleVisShader->setUniform("m",  bundle.model);
        _moduleVisShader->setUniform("isSelected",  !settings.hideSelectedModuleHighlight && settings.selectedModuleId == moduleID);
        _moduleVisShader->setUniform("warningFlag",  bundle.warning);
        _moduleVisShader->setUniform("propType",  settings.moduleVisualizationMode);

        if (settings.moduleVisualizationMode == MODULE_TEMPERATURE){
            _moduleVisShader->setUniform("propMax",  settings.visualizeForestVoxelGridMaxTemp);
            _moduleVisShader->setUniform("propMin",  settings.visualizeForestVoxelGridMinTemp);
            double temp = module->getCurrentState()->temperature;
            if (std::isnan(temp)) {
                _moduleVisShader->setUniform("warningFlag",  true);
                //Uncomment this if you want the simulation to pause the simulation
                //once you start getting bad values....
                settings.simulatorTimescale = 0;
            }
            _moduleVisShader->setUniform("prop", (float) temp);
        }

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
        _moduleVisShader->applyMaterial(mat);
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

void BasicForestScene::setLights(CS123Shader *selectedShader)
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        selectedShader->setLight(lightingInformation[i]);
    }
}

void BasicForestScene::setGlobalData(CS123Shader *selectedShader){
    selectedShader->setUniform("ka", globalData.ka);
    selectedShader->setUniform("kd", globalData.kd);
    selectedShader->setUniform("ks", globalData.ks);
}

void BasicForestScene::setSceneUniforms(SupportCanvas3D *context, CS123Shader *selectedShader) {
    Camera *camera = context->getCamera();
    selectedShader->setUniform("useLighting", settings.useLighting);
    selectedShader->setUniform("p" , camera->getProjectionMatrix());
    selectedShader->setUniform("v", camera->getViewMatrix());
}

void BasicForestScene::settingsChanged() {
     _voxelGrid.getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
     _voxelGrid.getVisualization()->updateValuesFromSettings();
}

Forest * BasicForestScene::getForest(){
    return _forest.get();
}

VoxelGrid * BasicForestScene::getVoxelGrid(){
    return &_voxelGrid;
}
