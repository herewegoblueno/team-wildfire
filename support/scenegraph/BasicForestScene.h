#ifndef BASICFORESTSCENE_H
#define BASICFORESTSCENE_H


#include "OpenGLScene.h"

#include <memory>
#include <vector>
#include "support/shapes/Shape.h"
#include "trees/forest.h"
#include "support/shapes/Trunk.h"
#include "support/shapes/Leaf.h"
#include "simulation/simulator.h"
#include "fire/firemanager.h"
#include <unordered_map>


const int gridBuffer = 3; // make grid slightly larger than forest
const int numTrees = 20;
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

    void renderTrunksVisualizedModules();
    void renderTrunks();
    void renderLeaves();
    void renderGeometry();
    void loadShaders();
    void tessellateShapes();

    void defineLights();
    void defineGlobalData();
    void setLights(CS123::GL::CS123Shader *s);
    void setGlobalData(CS123::GL::CS123Shader *s);
    void setSceneUniforms(SupportCanvas3D *context, CS123::GL::CS123Shader *s);

    std::unique_ptr<CS123::GL::CS123Shader> _phongShader;
    std::unique_ptr<CS123::GL::CS123Shader> _moduleVisShader;
    VoxelGrid _voxelGrid;
    std::unique_ptr<Forest> _forest;
    std::unique_ptr<Trunk> _trunk;
    std::unique_ptr<Leaf> _leaf;
    std::vector<PrimitiveBundle> _trunks;
    std::vector<PrimitiveBundle> _leaves;
    std::unordered_map<int, CS123SceneMaterial> _moduleIDToMat;
    Simulator _simulator;
    FireManager _fire_mngr;

    MainWindow *mainWindow;
};


#endif // BASICFORESTSCENE_H
