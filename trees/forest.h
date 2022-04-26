#ifndef FOREST_H
#define FOREST_H

#include "TreeGenerator.h"
#include "voxels/voxelgrid.h"
#include "unordered_map"

// Size of grid to search beyond center voxel when finding voxels
// that overlap a module
const int voxelSearchRadius = 25;

class Forest
{
public:
    Forest(VoxelGrid *grid, int numTrees, float forestWidth, float forestHeight);
    ~Forest();
    void update();
    std::vector<PrimitiveBundle> getPrimitives();
    void connectModulesToVoxels(VoxelGrid *voxelGrid);
    VoxelSet getVoxelsMappedToModule(Module *m);
    Module *getModuleFromId(int id);
    std::vector<int> getAllModuleIDs();

private:
    void createTrees(int numTrees, float forestWidth, float forestHeight);
    void addTreeToForest(const ModuleSet &modules, glm::mat4 trans);
    void initializeTrunkPrimitive();
    void initializeLeafPrimitive();

    VoxelGrid *_grid;
    BranchSet _branches;
    ModuleSet _modules;
    std::vector<PrimitiveBundle> _primitives;
    std::unique_ptr<TreeGenerator> _treeGenerator;
    std::unique_ptr<CS123ScenePrimitive> _trunk;
    std::unique_ptr<CS123ScenePrimitive> _leaf;

    void checkModuleVoxelOverlap(Module *module, Voxel *voxel, double cellSideLength);
    void connectModulesToVoxels();
    void initMassOfModules();
    void initMassOfVoxels();
    void updateVoxelMass(Voxel *voxel, double cellSideLength);
    std::map<Module *, VoxelSet> _moduleToVoxels;
    std::map<Voxel *, ModuleSet> _voxelToModules;
    std::unordered_map<int, Module *> _moduleIDs;
};

#endif // FOREST_H
