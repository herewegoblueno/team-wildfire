#ifndef BASICFORESTSCENE_H
#define BASICFORESTSCENE_H


#include "OpenGLScene.h"

#include <memory>
#include <vector>
#include "support/shapes/Shape.h"
#include "voxels/voxelgrid.h"
#include "trees/forest.h"
#include "support/shapes/Trunk.h"
#include "support/shapes/Leaf.h"
#include "simulation/simulator.h"

const int numTrees = 20;
const float forestHeight = 10;
const float forestWidth = 10;

namespace CS123 { namespace GL {
    class Shader;
    class CS123Shader;
    class Texture2D;
}}


class BasicForestScene : public OpenGLScene {
public:
    BasicForestScene();
    virtual ~BasicForestScene();

    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

private:
    void updateFromForest();

    void renderTrunks();
    void renderLeaves();
    void renderGeometry();
    void loadPhongShader();
    void tessellateShapes();

    void defineLights();
    void defineGlobalData();
    void setLights();
    void setGlobalData();
    void setSceneUniforms(SupportCanvas3D *context);

    std::unique_ptr<CS123::GL::CS123Shader> _phongShader;
    VoxelGrid _voxelGrids;
    std::unique_ptr<Forest> _forest;
    std::unique_ptr<Trunk> _trunk;
    std::unique_ptr<Leaf> _leaf;
    std::vector<PrimitiveBundle> _trunks;
    std::vector<PrimitiveBundle> _leaves;
    Simulator _simulator;
};


#endif // BASICFORESTSCENE_H
