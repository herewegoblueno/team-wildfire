#include "ForestScene.h"
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

ForestScene::ForestScene(MainWindow *mainWindow):
     mainWindow(mainWindow)
{
    loadShaders();
    tessellateShapes();
}

ForestScene::~ForestScene()
{
}

/** Initialize the scene after it has been filled by the parser */
void ForestScene::init() {
    VoxelGridDim gridDimensions = computeGridDimensions();
    _voxelGrid = std::make_unique<VoxelGrid>(gridDimensions, 64);
    _fireManager = std::make_unique<FireManager>(_voxelGrid.get()),
    _voxelGrid->getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
    _forest = std::make_unique<Forest>(_voxelGrid.get(), _fireManager.get(),
                                       treeRegions, gridDimensions);
    _lastFrameNumModules = _forest->getAllModuleIDs().size();
    //The forest also initializes the mass of the voxels
    updatePrimitivesFromForest();
    _simulator = std::make_unique<Simulator>();
    _simulator->init();
    _voxelGrid->getVisualization()->setForestReference(_forest.get());
    mainWindow->updateModuleSelectionOptions(_forest->getAllModuleIDs());
}

/**
 * Update fires for any modules that started or stopped burning since previous frame
 */
void ForestScene::updateFires() {
    for (Module *m : _forest->getModules()) {
        bool burningLastFrame = _lastFrameModuleBurnState[m];
        double massChangeRate = m->getCurrentState()->massChangeRateFromLastFrame;
        bool burningThisFrame = massChangeRate < 0.0;
        if (burningThisFrame) {
            // Adjust size of fire based on mass change rate
            float fireSize = _fireManager->massChangeRateToFireSize(massChangeRate);
            if (!burningLastFrame) {
                for (vec3 &fireSpawnPos : m->_fireSpawnPoints) {
                    _fireManager->addFire(m, fireSpawnPos, fireSize);
                }
            } else {
                _fireManager->setModuleFireSizes(m, fireSize);
            }
        } else if (burningLastFrame) {
            _fireManager->removeFires(m);
        }
        _lastFrameModuleBurnState[m] = burningThisFrame;
    }
}

/** Update scene primitives based on the new state of the forest */
void ForestScene::updatePrimitivesFromForest() {
    _leafBundles.clear();
    _trunkBundles.clear();
    _forest->recalculatePrimitives();
    std::vector<PrimitiveBundle> forestPrimitives = _forest->getPrimitives();
    PrimitiveType type;
    for (PrimitiveBundle &bundle : forestPrimitives) {
        type = bundle.primitive.type;
        if (type == PrimitiveType::PRIMITIVE_TRUNK) {
            _trunkBundles.push_back(bundle);
        } else if (type == PrimitiveType::PRIMITIVE_LEAF) {
            _leafBundles.push_back(bundle);
        } else if (type == PrimitiveType::PRIMITIVE_GROUND) {
            _groundBundle = bundle;
        }
    }
}

void ForestScene::loadShaders() {
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
    _phongShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);

    fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/module.frag");
    _moduleVisShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}

void ForestScene::render(SupportCanvas3D *context) {
    _simulator->step(_voxelGrid.get(), _forest.get());

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

    _voxelGrid->getVisualization()->setPV(camera->getProjectionMatrix() * camera->getViewMatrix());
    _voxelGrid->getVisualization()->draw(context);

    updateFires();
    _fireManager->setCamera(camera->getProjectionMatrix(), camera->getViewMatrix());
    _fireManager->setScale(0.03, 0.05);
    _fireManager->drawFires(_simulator->getTimeSinceLastFrame()/1000.0, true);

    _simulator->cleanupForNextStep(_voxelGrid.get(), _forest.get());

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
void ForestScene::tessellateShapes() {
    _leaf = std::make_unique<Leaf>();
    _trunk = std::make_unique<Trunk>(1, 1);
    _ground = std::make_unique<Ground>();
}

/** Render each primitive's shape */
void ForestScene::renderGeometry() {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    renderGround();
    if (settings.seeBranchModules) {
        renderTrunksVisualizedModules();
    } else {
        renderTrunks();
        renderLeaves();
    }
}

/** Color each branch based on the module it's in */
void ForestScene::renderTrunksVisualizedModules() {
    _trunk->bindVAO();
    for (PrimitiveBundle &bundle : _trunkBundles) {
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

void ForestScene::renderTrunks() {
    _trunk->bindVAO();
    for (PrimitiveBundle &bundle : _trunkBundles) {
        _phongShader->setUniform("m",  bundle.model);
        _phongShader->applyMaterial(bundle.primitive.material);
        _trunk->drawVAO();
    }
    _trunk->unbindVAO();
}

void ForestScene::renderLeaves() {
    _leaf->bindVAO();
    for (PrimitiveBundle &bundle : _leafBundles) {
        _phongShader->setUniform("m",  bundle.model);
        _phongShader->applyMaterial(bundle.primitive.material);
        _leaf->drawVAO();
    }
    _leaf->unbindVAO();
}

void ForestScene::renderGround() {
    _ground->bindVAO();
    _phongShader->setUniform("m",  _groundBundle.model);
    _phongShader->applyMaterial(_groundBundle.primitive.material);
    _ground->drawVAO();
    _ground->unbindVAO();
}

/** Compute center and axis size of grid such that it bounds all tree regions */
VoxelGridDim ForestScene::computeGridDimensions() {
    float minX = FLT_MAX;
    float minZ = FLT_MAX;
    float maxX = -FLT_MAX;
    float maxZ = -FLT_MAX;
    for (TreeRegionData &region : treeRegions) {
        vec3 center = region.center;
        float width = region.width;
        float height = region.height;
        float regionMinX = (center.x - width / 2.f);
        float regionMinZ = (center.z - height / 2.f);
        float regionMaxX = (center.x + width / 2.f);
        float regionMaxZ = (center.z + height / 2.f);
        minX = std::min(minX, regionMinX);
        minZ = std::min(minZ, regionMinZ);
        maxX = std::max(maxX, regionMaxX);
        maxZ = std::max(maxZ, regionMaxZ);
    }
    float width = maxX - minX;
    float height = maxZ - minZ;
    float axisSize = std::max(width, height) + gridBuffer;
    // shift up by axisSize / 2 so the bottom of the grid is at y=0
    vec3 center = vec3(minX + width / 2.f, axisSize / 2.f, minZ + height / 2.f);
    return VoxelGridDim(center, axisSize);
}

void ForestScene::setLights(CS123Shader *selectedShader)
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        selectedShader->setLight(lightingInformation[i]);
    }
}

void ForestScene::setGlobalData(CS123Shader *selectedShader){
    selectedShader->setUniform("ka", globalData.ka);
    selectedShader->setUniform("kd", globalData.kd);
    selectedShader->setUniform("ks", globalData.ks);
}

void ForestScene::setSceneUniforms(SupportCanvas3D *context, CS123Shader *selectedShader) {
    Camera *camera = context->getCamera();
    selectedShader->setUniform("useLighting", settings.useLighting);
    selectedShader->setUniform("p" , camera->getProjectionMatrix());
    selectedShader->setUniform("v", camera->getViewMatrix());
}

void ForestScene::settingsChanged() {
     _voxelGrid->getVisualization()->toggle(settings.visualizeForestVoxelGrid, settings.visualizeVectorField);
     _voxelGrid->getVisualization()->updateValuesFromSettings();
}

Forest * ForestScene::getForest(){
    return _forest.get();
}

VoxelGrid * ForestScene::getVoxelGrid(){
    return _voxelGrid.get();
}
