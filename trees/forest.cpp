#include "forest.h"
#include "glm/gtx/transform.hpp"
#include "glm/gtx/closest_point.hpp"
#include <iostream>

using namespace glm;

Forest::Forest(VoxelGrid *grid, int numTrees, float forestWidth, float forestHeight) :
    _grid(grid),
    _treeGenerator(nullptr)
{
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
    createTrees(numTrees, forestWidth, forestHeight);
    initMassAndAreaOfModules();
    initializeModuleVoxelMapping(); // depends on module mass
    initMassOfVoxels(); // depends on voxel mapping
    initTempOfModules(); // depends on voxel mapping
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
        Module *m = getModuleFromId(branch->moduleID);
        mat4 model = branch->model * glm::scale(glm::vec3(m->getCurrentState()->radiusRatio,
                                                          1.0f, m->getCurrentState()->radiusRatio));
        PrimitiveBundle branchPrimitive(*_trunk, model, branch->moduleID);
        if (m->_warning) {
            branchPrimitive.warning = true;
        }
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
        _moduleIDs[module->ID] = module;
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

   //Rest of this is just for debugging output...
   int totalVoxels = 0;
   for (Module *module : _modules) {
       int numVoxels = _moduleToVoxels[module].size();
       totalVoxels += numVoxels;
       if (numVoxels == 0) {
           std::cerr << "Module " << module->ID << " has 0 voxels "<<  std::endl;
           module->_warning = true;
       }
   }
   std::cout << (float)totalVoxels/(float)_modules.size() << " voxels per module" << std::endl;

   int totalModules = 0;
   for (auto const& voxelModules : _voxelToModules) {
       totalModules += voxelModules.second.size();
   }
   std::cout << (float)totalModules/(float)(std::pow(resolution,3)) << " modules per voxel" << std::endl;
}

//Since trees can only shrink over time, we can only ever lose voxels in the mappings...
void Forest::updateModuleVoxelMapping(){
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

/** See if a module and voxel overlap by checking each branch */
bool Forest::checkModuleVoxelOverlap(Module *module, Voxel *voxel,
                                     double cellSideLength) {
    dvec3 voxelCenter = voxel->centerInWorldSpace;
    for (Branch *branch: module->_branches) {
        //This is called in updateModuleVoxelMapping, which is called before updateLastFrameDataOfModules in the
        //simulation loop, so we should use getCurrentState
        double ratio = module->getCurrentState()->radiusRatio;
        mat4 model = branch->model;
        double branchRadius = branch->radius;
        double branchLength = branch->length;
        double cellRadius = cellSideLength / 2.0;
        dvec3 branchStart = dvec3(model * trunkObjectBottom);
        dvec3 branchEnd = dvec3(model * trunkObjectTop);
        // Ensure lengths are consistent
        assert(abs(branchLength - length(branchEnd - branchStart)) < 0.0001);
        // Check branch start / end boundary
        dvec3 toVoxelCenter = voxelCenter - branchStart;
        dvec3 branchDir = normalize(branchEnd - branchStart);
        double scalarProj = dot(toVoxelCenter, branchDir);
        if (scalarProj + cellRadius < 0.0) {
            continue;
        }
        if (scalarProj - cellRadius > branchLength) {
            continue;
        }
        // Check branch radius boundary
        scalarProj = std::min(std::max(scalarProj, 0.0), branchLength);
        dvec3 closestPoint = branchStart + branchDir * scalarProj;
        double distUpBranch = scalarProj / branchLength;
        assert(distUpBranch <= 1.0 + DBL_EPSILON);
        double horizScale = (1.0 - (1.0 - branchWidthDecay) * distUpBranch) * ratio;
        assert(horizScale <= 1.0 + DBL_EPSILON);
        if (length(closestPoint - voxelCenter) < branchRadius * horizScale + cellRadius) {
            return true;
        }
    }
    return false;
}
/** Init temp of each module to average of surrounding voxel ambient temps */
void Forest::initTempOfModules() {
    for (Module *module : _modules) {
        double totalTemp = 0;
        VoxelSet voxels = _moduleToVoxels[module];
        for (Voxel *vox : voxels) {
            totalTemp += vox->getAmbientTemperature();
        }
        double numVoxels = static_cast<double>(voxels.size());
        module->getCurrentState()->temperature = totalTemp / numVoxels;
        module->getLastFrameState()->temperature = totalTemp / numVoxels;
    }
}

/** Init mass of each module based on its branches */
void Forest::initMassAndAreaOfModules() {
    for (Module *module : _modules) {
        module->initMassAndArea();
    }
}

void Forest::updateMassAndAreaOfModules(){
    for (Module *module : _modules) {
        module->updateMassAndArea();
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

void Forest::updateMassOfVoxels(){
    for (auto const& moduleToVoxels : _moduleToVoxels) {
        Module *module = moduleToVoxels.first;
        VoxelSet voxels = moduleToVoxels.second;
        double numVoxels = voxels.size();
        double massChangePerVoxel = (module->getLastFrameState()->mass - module->getCurrentState()->mass) / numVoxels;
        for (Voxel *voxel : voxels) {
            voxel->getCurrentState()->mass -= massChangePerVoxel;
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

void Forest::deleteModuleAndChildren(Module *m){
    //First remove it from all the necessary maps
    //TODO: there might be a tiny amount of mass left from that module in the associated voxels, maybe we should eventually clean that up
    _moduleIDs.erase(m->ID);

    VoxelSet associatedVoxels = getVoxelsMappedToModule(m);
    for (Voxel *v : associatedVoxels) _voxelToModules[v].erase(m);
    _moduleToVoxels.erase(m);

    if (m->_parent != nullptr) m->_parent->_children.erase(m);

    for (Branch *b : m->_branches) {
        _branches.erase(b);
        delete b;
    }

    //Copying the original children, since deleteModuleAndChildren will edit _children of this module
    ModuleSet originalChildren = m->_children;
    for (Module *child : originalChildren) {
        deleteModuleAndChildren(child);
    }

    _modules.erase(m);
    delete m; //i bid you adieu
}


void Forest::deleteDeadModules(){
    //This is called before updateLastFrameDataOfModules, so we should use currentState
    //Not using a normal iterator since I'm not sure the order of appearance of parents and children
    //so you don't know where to reposition the interator once one call to deleteModuleAndChildren has completed
    //Also, since unordered_maps seem to be able to rehash(https://stackoverflow.com/questions/18301302/is-forauto-i-unordered-map-guaranteed-to-have-the-same-order-every-time)
    //it might not be safe for deleteModuleAndChildren to return the left-most iterator of all the deletes
    for(Module *m : _modules)
    {
       if (m->getCurrentState()->mass <= 0){
           deleteModuleAndChildren(m);
           return deleteDeadModules();
       }
    }
}
