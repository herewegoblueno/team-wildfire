#ifndef FOREST_H
#define FOREST_H

#include "TreeGenerator.h"
#include "voxels/voxelgrid.h"

// Size of grid to search beyond center voxel when finding voxels
// that overlap a module
const int voxelSearchRadius = 25;

class Forest
{
public:
    Forest(int numTrees, float forestWidth, float forestHeight);
    ~Forest();
    void update();
    std::vector<PrimitiveBundle> getPrimitives();
    void connectModulesToVoxels(VoxelGrid *voxelGrid);

private:
    void createTrees(int numTrees, float forestWidth, float forestHeight);
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

    void checkModuleVoxelOverlap(Module *module, Voxel *voxel,
                                 double cellSideLength);
    std::map<Module *, VoxelSet> _moduleToVoxels;
    std::map<Voxel *, ModuleSet> _voxelToModules;
};

#endif // FOREST_H
