#ifndef FOREST_H
#define FOREST_H

#include "TreeGenerator.h"
#include "voxels/voxelgrid.h"
#include "unordered_map"
#include "fire/firemanager.h"

const int gridBuffer = 3; // make grid slightly larger than forest
// Size of grid to search beyond center voxel when finding voxels
// that overlap a module
const int voxelSearchRadius = 25;

class Forest
{
public:
    Forest(VoxelGrid *grid, FireManager *fireManager,
           int numTrees, float forestWidth, float forestHeight);
    ~Forest();
    void recalculatePrimitives();
    std::vector<PrimitiveBundle> getPrimitives();

    void updateModuleVoxelMapping();
    void updateMassAndAreaOfModulesViaBurning(double deltaTimeInMs);
    void updateMassOfVoxels();
    void artificiallyUpdateTemperatureOfModule(int moduleID, double delta);
    void artificiallyUpdateVoxelTemperatureAroundModule(int moduleID, double delta);

    ModuleSet getModulesMappedToVoxel(Voxel *v);
    VoxelSet getVoxelsMappedToModule(Module *m);
    Module *getModuleFromId(int id);
    std::vector<int> getAllModuleIDs();
    ModuleSet getModules();

    void updateLastFrameDataOfModules();
    void deleteDeadModules();

private:
    void createTrees();
    void addTreeToForest(const ModuleSet &modules, glm::mat4 trans);
    void initializeTrunkPrimitive();
    void initializeLeafPrimitive();
    void initializeGroundPrimitive();

    int _numTrees;
    float _forestWidth;
    float _forestHeight;
    FireManager *_fireManager;
    VoxelGrid *_grid;
    BranchSet _branches;
    ModuleSet _modules;
    std::vector<PrimitiveBundle> _primitives;
    std::unique_ptr<TreeGenerator> _treeGenerator;
    std::unique_ptr<CS123ScenePrimitive> _trunk;
    std::unique_ptr<CS123ScenePrimitive> _leaf;
    std::unique_ptr<CS123ScenePrimitive> _ground;
    glm::mat4 _groundModel;

    bool checkModuleVoxelOverlap(Module *module, Voxel *voxel, double cellSideLength);
    void initializeModuleVoxelMapping();
    void connectModulesToVoxels();
    void initTempOfModules();
    void initModuleProperties();
    void initMassOfVoxels();
    std::map<Module *, VoxelSet> _moduleToVoxels;
    std::map<Voxel *, ModuleSet> _voxelToModules;
    std::unordered_map<int, Module *> _moduleIDs;

    void deleteModuleAndChildren(Module *m);
};

#endif // FOREST_H
