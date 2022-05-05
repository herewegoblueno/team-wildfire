#ifndef BASICFORESTSCENE_H
#define BASICFORESTSCENE_H


#include "OpenGLScene.h"

#include <map>
#include <memory>
#include <vector>
#include "trees/forest.h"
#include "support/shapes/Shape.h"
#include "support/shapes/Trunk.h"
#include "support/shapes/Leaf.h"
#include "support/shapes/Ground.h"
#include "simulation/simulator.h"
#include <unordered_map>

const int numTrees = 100;
const float forestHeight = 10;
const float forestWidth = 10;

class BasicForestScene : public OpenGLScene {
public:
    BasicForestScene(MainWindow *mainWindow);
    virtual ~BasicForestScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

    Forest *getForest();
    VoxelGrid *getVoxelGrid();

private:
    void updatePrimitivesFromForest();
    void updateFires();

    void renderTrunksVisualizedModules();
    void renderTrunks();
    void renderLeaves();
    void renderGround();
    void renderGeometry();
    void loadShaders();
    void tessellateShapes();

    void defineLights();
    void defineGlobalData();
    void setLights(CS123::GL::CS123Shader *s);
    void setGlobalData(CS123::GL::CS123Shader *s);
    void setSceneUniforms(SupportCanvas3D *context, CS123::GL::CS123Shader *s);

    std::map<Module *, bool> _lastFrameModuleBurnState;
    uint _lastFrameNumModules;
    std::unique_ptr<CS123::GL::CS123Shader> _phongShader;
    std::unique_ptr<CS123::GL::CS123Shader> _moduleVisShader;
    VoxelGrid _voxelGrid;
    std::unique_ptr<Forest> _forest;
    std::unique_ptr<Trunk> _trunk;
    std::unique_ptr<Leaf> _leaf;
    std::unique_ptr<Ground> _ground;
    std::vector<PrimitiveBundle> _trunkBundles;
    std::vector<PrimitiveBundle> _leafBundles;
    PrimitiveBundle _groundBundle;
    std::unordered_map<int, CS123SceneMaterial> _moduleIDToMat;
    Simulator _simulator;
    FireManager _fireManager;

    MainWindow *mainWindow;
};


#endif // BASICFORESTSCENE_H
