#ifndef FOREST_H
#define FOREST_H

#include "TreeGenerator.h"

class Forest
{
public:
    Forest(int numTrees, float forestWidth, float forestHeight);
    std::vector<PrimitiveBundle> getPrimitives();

private:
    void addPrimitivesFromBranches(const BranchSet &branches, glm::mat4 trans);
    void initializeTrunkPrimitive();
    void initializeLeafPrimitive();

    std::vector<PrimitiveBundle> m_primitives;
    std::unique_ptr<TreeGenerator> m_treeGenerator;
    std::unique_ptr<CS123ScenePrimitive> m_trunk;
    std::unique_ptr<CS123ScenePrimitive> m_leaf;
};

#endif // FOREST_H
