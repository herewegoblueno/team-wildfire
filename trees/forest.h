#ifndef FOREST_H
#define FOREST_H

#include "TreeGenerator.h"

class Forest
{
public:
    Forest(int numTrees, float forestWidth, float forestHeight);
    ~Forest();
    void update();
    std::vector<PrimitiveBundle> getPrimitives();

private:
    void addTreeToForest(const ModuleSet &modules, glm::mat4 trans);
    void initializeTrunkPrimitive();
    void initializeLeafPrimitive();

    BranchSet _branches;
    ModuleSet _modules;
    std::vector<PrimitiveBundle> _primitives;
    std::unique_ptr<TreeGenerator> _treeGenerator;
    std::unique_ptr<CS123ScenePrimitive> _trunk;
    std::unique_ptr<CS123ScenePrimitive> _leaf;
    int _moduleNum;
};

#endif // FOREST_H
