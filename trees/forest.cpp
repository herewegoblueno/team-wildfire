#include "forest.h"
#include "glm/gtx/transform.hpp"
#include <iostream>

using namespace glm;

Forest::Forest(int numTrees, float forestWidth, float forestHeight) :
    _treeGenerator(nullptr),
    _moduleNum(0)
{
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
    createTrees(numTrees, forestWidth, forestHeight);
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
void Forest::update() {
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
        _modules.insert(module);
        _moduleNum++;
        for (Branch *branch : module->_branches) {
            if (seen.count(branch)) {
                std::cerr << "ERROR: BRANCH IN MULTIPLE MODULES" << std::endl;
            }
            seen.insert(branch);
            branch->moduleID = _moduleNum;
            branch->model = trans * branch->model;
            _branches.insert(branch);
            PrimitiveBundle branchPrimitive(*_trunk, branch->model, _moduleNum);
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
void Forest::connectModulesToVoxels(VoxelGrid *grid) {
   int resolution = grid->getResolution();
   double cellSideLength = grid->cellSideLength();
   for (Module *module: _modules) {
       vec3 centerPos = vec3(module->getCenter());
       Voxel *center = grid->getVoxelClosestToPoint(centerPos);
       int xMin = std::max(0, center->XIndex - voxelSearchRadius);
       int xMax = std::min(resolution, center->XIndex + voxelSearchRadius);
       int yMin = std::max(0, center->YIndex - voxelSearchRadius);
       int yMax = std::min(resolution, center->YIndex + voxelSearchRadius);
       int zMin = std::max(0, center->ZIndex - voxelSearchRadius);
       int zMax = std::min(resolution, center->ZIndex + voxelSearchRadius);
       for (int x = xMin; x < xMax; x++) {
           for (int y = yMin; y < yMax; y++) {
               for (int z = zMin; z < zMax; z++) {
                   Voxel *voxel = grid->getVoxel(x, y, z);
                   checkModuleVoxelOverlap(module, voxel, cellSideLength);
               }
           }
       }
   }

   int total = 0;
   for (auto const& moduleVoxels : _moduleToVoxels) {
       total += moduleVoxels.second.size();
   }
   std::cout << (float)total/(float)_modules.size() << " voxels per module" << std::endl;
}

/** See if a module and voxel overlap by checking each branch */
void Forest::checkModuleVoxelOverlap(Module *module, Voxel *voxel,
                                     double cellSideLength) {
    vec3 voxelCenter = voxel->centerInWorldSpace;
    for (Branch *branch: module->_branches) {
        vec4 branchSpaceCenter = branch->invModel * vec4(voxelCenter, 1);
        float x = branchSpaceCenter.x;
        float y = branchSpaceCenter.y;
        float z = branchSpaceCenter.z;
        double dist = std::sqrt(x*x + z*z); // lateral dist to branch center
        // implicit branch boundary
        double branchMaxDist = trunkInitRadius *
                branchWidthDecay * (y / trunkInitLength);
        // approximate voxel as a sphere
        if (dist - cellSideLength < branchMaxDist) {
            _moduleToVoxels[module].insert(voxel);
            _voxelToModules[voxel].insert(module);
            return;
        }
    }
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
    _leaf = std::make_unique<CS123ScenePrimitive>(
                PrimitiveType::PRIMITIVE_LEAF, *material);
}
