#include "forest.h"
#include "glm/gtx/transform.hpp"
#include <iostream>

using namespace glm;

Forest::Forest(VoxelGrid *grid, int numTrees, float forestWidth, float forestHeight) :
    _grid(grid),
    _treeGenerator(nullptr)
{
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
    createTrees(numTrees, forestWidth, forestHeight);
    initMassOfModules();
    initializeModuleVoxelMapping(); // depends on module mass
    initMassOfVoxels();
}

Forest::~Forest() {
    for (Branch *branch : _branches) {
        delete branch;
    }
    for (Module *module : _modules) {
        delete module;
    }
}

/** Update primitives based on branches */
void Forest::recalculatePrimitives() {
    _primitives.clear();
    for (Branch *branch : _branches) {
        PrimitiveBundle branchPrimitive(*_trunk, branch->model, branch->moduleID);
        _primitives.push_back(branchPrimitive);
        for (mat4 &leafModel : branch->leafModels) {
            PrimitiveBundle leafPrimitive(*_leaf, leafModel);
            _primitives.push_back(leafPrimitive);
        }
    }
}

/** Generate trees, add their modules and branches to state */
void Forest::createTrees(int numTrees, float forestWidth, float forestHeight) {
    _treeGenerator = std::make_unique<TreeGenerator>();
    int totalModules = 0;
    for (int i = 0; i < numTrees; i++) {
        float x = randomFloat() * forestWidth - forestWidth / 2;
        float z = randomFloat() * forestHeight - forestHeight / 2;
        mat4 trans = translate(vec3(x, 0, z));
        _treeGenerator->generateTree();
        ModuleTree moduleTree = _treeGenerator->getModuleTree();
        addTreeToForest(moduleTree.modules, trans);
        totalModules += moduleTree.modules.size();
    }
    std::cout << (float)totalModules/(float)numTrees << " modules per tree" << std::endl;
}

/**
 * Add modules and branches to forest state, adjusted with a transformation
 * to get the tree in the desired position
 */
void Forest::addTreeToForest(const ModuleSet &modules, mat4 trans) {
    std::unordered_set<Branch *> seen;
    for (Module *module : modules) {
        _moduleIDs.insert({module->ID, module});
        _modules.insert(module);
        for (Branch *branch : module->_branches) {
            if (seen.count(branch)) {
                std::cerr << "ERROR: BRANCH IN MULTIPLE MODULES" << std::endl;
            }
            seen.insert(branch);
            branch->moduleID = module->ID;
            branch->model = trans * branch->model;
            branch->invModel = inverse(branch->model);
            _branches.insert(branch);
            PrimitiveBundle branchPrimitive(*_trunk, branch->model, module->ID);
            _primitives.push_back(branchPrimitive);
            for (mat4 &leafModel : branch->leafModels) {
                leafModel = trans * leafModel;
                PrimitiveBundle leafPrimitive(*_leaf, leafModel);
                _primitives.push_back(leafPrimitive);
            }
        }
    }
}

/** Map modules to voxels and vice versa */
void Forest::initializeModuleVoxelMapping() {
   int resolution = _grid->getResolution();
   double cellSideLength = _grid->cellSideLength();
   for (Module *module: _modules) {
       vec3 centerPos = vec3(module->getCenter());
       Voxel *center = _grid->getVoxelClosestToPoint(centerPos);
       int xMin = std::max(0, center->XIndex - voxelSearchRadius);
       int xMax = std::min(resolution, center->XIndex + voxelSearchRadius);
       int yMin = std::max(0, center->YIndex - voxelSearchRadius);
       int yMax = std::min(resolution, center->YIndex + voxelSearchRadius);
       int zMin = std::max(0, center->ZIndex - voxelSearchRadius);
       int zMax = std::min(resolution, center->ZIndex + voxelSearchRadius);
       for (int x = xMin; x < xMax; x++) {
           for (int y = yMin; y < yMax; y++) {
               for (int z = zMin; z < zMax; z++) {
                   Voxel *voxel = _grid->getVoxel(x, y, z);
                   if (checkModuleVoxelOverlap(module, voxel, cellSideLength)){
                       _moduleToVoxels[module].insert(voxel);
                       _voxelToModules[voxel].insert(module);
                   }
               }
           }
       }
   }

   int totalVoxels = 0;
   for (auto const& moduleVoxels : _moduleToVoxels) {
       int moduleID = moduleVoxels.first->ID;
       int numVoxels = moduleVoxels.second.size();
       totalVoxels += numVoxels;

       if (numVoxels == 0) {
           std::cerr << "Module " << moduleID << "has 0 voxels "<<  std::endl;
       }
   }
   std::cout << (float)totalVoxels/(float)_modules.size() << " voxels per module" << std::endl;

   int totalModules = 0;
   for (auto const& voxelModules : _voxelToModules) {
       totalModules += voxelModules.second.size();
   }
   std::cout << (float)totalModules/(float)(std::pow(resolution,3)) << " modules per voxel" << std::endl;
}

/** See if a module and voxel overlap by checking each branch */
bool Forest::checkModuleVoxelOverlap(Module *module, Voxel *voxel,
                                     double cellSideLength) {
    dvec3 voxelCenter = voxel->centerInWorldSpace;
    for (Branch *branch: module->_branches) {
        vec4 branchSpaceCenter = branch->invModel * vec4(voxelCenter, 1);
        float x = branchSpaceCenter.x;
        float y = branchSpaceCenter.y;
        float z = branchSpaceCenter.z;
        // lateral dist to branch center
        double dist = std::sqrt(x*x + z*z);
        // implicit lateral branch boundary
        double horizScale = 1.0 - (1.0 - branchWidthDecay) * (y + 1.0);
        double branchMaxDist = 0.5 * horizScale;
        // approximate voxel as a sphere
        if (dist - cellSideLength / 2.0 < branchMaxDist && y >= -0.5 && y <= 0.5) {
            return true;
        }
    }
    return false;
}

/** Init mass of each module based on its branches */
void Forest::initMassOfModules() {
    for (Module *module : _modules) {
        module->initMass();
    }
}

void Forest::updateMassOfModules(){
    for (Module *module : _modules) {
        module->updateMass();
    }
}

/**
 * Update the structs that contain the info from last frame with the current frame's data (in preparation for another simulation run)
 */
void Forest::updateLastFrameDataOfModules(){
    for (Module *module : _modules) {
        module->updateLastFrameData();
    }
}

/**
 * Evenly distribute mass of each module over all the voxels it overlaps
 */
void Forest::initMassOfVoxels() {
    for (auto const& moduleToVoxels : _moduleToVoxels) {
        Module *module = moduleToVoxels.first;
        VoxelSet voxels = moduleToVoxels.second;
        double numVoxels = voxels.size();
        double massPerVoxel = module->getCurrentState()->mass / numVoxels;
        for (Voxel *voxel : voxels) {
            voxel->getLastFrameState()->mass += massPerVoxel;
            voxel->getCurrentState()->mass += massPerVoxel;
        }
    }
}

void Forest::updateModuleVoxelMapping(VoxelGrid *voxelGrid){
    double cellSideLength = _grid->cellSideLength();

    for (auto const& moduleToVoxels : _moduleToVoxels) {
        Module *module = moduleToVoxels.first;
        VoxelSet voxels = moduleToVoxels.second;
        for (Voxel *voxel : voxels) {
            if (!checkModuleVoxelOverlap(module, voxel, cellSideLength)){
                _moduleToVoxels[module].erase(voxel);
                _voxelToModules[voxel].erase(module);
            }
        }
    }
}

void updateMassOfVoxels(){
    //TODO
}

std::vector<PrimitiveBundle> Forest::getPrimitives() {
    return _primitives;
}

/** Initialize the cylinder building block of our tree trunk/branches */
void Forest::initializeTrunkPrimitive() {
    // Initialize brownish material for trunk
    std::unique_ptr<CS123SceneMaterial> material = std::make_unique<CS123SceneMaterial>();
    material->clear();
    material->cAmbient.r = 0.2f;
    material->cAmbient.g = 0.2f;
    material->cAmbient.b = 0.2f;
    material->cDiffuse.r = 0.4f;
    material->cDiffuse.g = 0.2f;
    material->cDiffuse.b = 0.2f;
    // Create primitive object
    _trunk = std::make_unique<CS123ScenePrimitive>(
                PrimitiveType::PRIMITIVE_TRUNK, *material);
}


/** Initialize the leaf primitive */
void Forest::initializeLeafPrimitive() {
    // Initialize green material for leaves
    std::unique_ptr<CS123SceneMaterial> material = std::make_unique<CS123SceneMaterial>();
    material->clear();
    material->cAmbient.r = 0.20f;
    material->cAmbient.g = 0.5f;
    material->cAmbient.b = 0.02f;
    material->cDiffuse.r = 0.20f;
    material->cDiffuse.g = 0.5f;
    material->cDiffuse.b = 0.02f;
    // Create primitive object
    _leaf = std::make_unique<CS123ScenePrimitive>(PrimitiveType::PRIMITIVE_LEAF, *material);
}


VoxelSet Forest::getVoxelsMappedToModule(Module *m){
    return _moduleToVoxels[m];
}

Module *Forest::getModuleFromId(int id){
    return _moduleIDs[id];
}

std::vector<int> Forest::getAllModuleIDs(){
    std::vector<int> keys;
    for(auto const& pair: _moduleIDs)
        keys.push_back(pair.first);
    return keys;
}
