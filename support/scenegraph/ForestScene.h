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
#include <QMouseEvent>

const int numTrees = 100;
const float forestHeight = 10;
const float forestWidth = 10;

class ForestScene : public OpenGLScene {
public:
    ForestScene(MainWindow *mainWindow);
    virtual ~ForestScene();

    void init();
    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

    Forest *getForest();
    VoxelGrid *getVoxelGrid();

    void onMousePress(QMouseEvent *event, SupportCanvas3D *canvas);

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

    VoxelGridDim computeGridDimensions();
    void setLights(CS123::GL::CS123Shader *s);
    void setGlobalData(CS123::GL::CS123Shader *s);
    void setSceneUniforms(SupportCanvas3D *context, CS123::GL::CS123Shader *s);

    std::map<Module *, bool> _lastFrameModuleBurnState;
    uint _lastFrameNumModules;
    std::unique_ptr<CS123::GL::CS123Shader> _phongShader;
    std::unique_ptr<CS123::GL::CS123Shader> _moduleVisShader;
    std::unique_ptr<Forest> _forest;
    std::unique_ptr<Trunk> _trunk;
    std::unique_ptr<Leaf> _leaf;
    std::unique_ptr<Ground> _ground;
    std::vector<PrimitiveBundle> _trunkBundles;
    std::vector<PrimitiveBundle> _leafBundles;
    PrimitiveBundle _groundBundle;
    std::unordered_map<int, CS123SceneMaterial> _moduleIDToMat;
    std::unique_ptr<VoxelGrid> _voxelGrid;
    std::unique_ptr<Simulator> _simulator;
    std::unique_ptr<FireManager> _fireManager;

    void changeTemperatureOfModulesAroundTemp(glm::vec3 center, double delta);

    MainWindow *mainWindow;
};


#endif // BASICFORESTSCENE_H
