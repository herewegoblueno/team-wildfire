#include "forest.h"
#include "glm/gtx/transform.hpp"
#include "glm/gtx/closest_point.hpp"
#include <iostream>

using namespace glm;

Forest::Forest(VoxelGrid *grid, FireManager *fireManager,
               TreeRegions regions, VoxelGridDim voxelGridDim) :
    _treeRegions(regions),
    _fireManager(fireManager),
    _grid(grid),
    _treeGenerator(nullptr),
    _voxelGridDim(voxelGridDim)
{
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
    initializeGroundPrimitive();
    for (TreeRegionData &region : _treeRegions) {
        createTrees(region);
    }
    initModuleProperties();
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
    PrimitiveBundle groundPrimitive(*_ground, _groundModel);
    _primitives.push_back(groundPrimitive);
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

/**
 * Generate trees, add their modules and branches to state.
 * Use stratified sampling to avoid tree overlap.
 */
void Forest::createTrees(TreeRegionData region) {
    glm::vec3 center = region.center;
    float width = region.width;
    float height = region.height;
    int numTrees = region.numTrees;
    _treeGenerator = std::make_unique<TreeGenerator>();
    int xStrata = std::max(static_cast<int>(std::sqrt(numTrees)), 1);
    int zStrata = numTrees / xStrata;
    float strataWidth = width / static_cast<float>(xStrata);
    float strataHeight = height / static_cast<float>(zStrata);
    for (int zStep = 0; zStep < zStrata; zStep++) {
        for (int xStep = 0; xStep < xStrata; xStep++) {
            // Sample a point in unit square
            float xSample = randomFloat();
            float zSample = randomFloat();
            // Map to a point on the current strata
            float x = (xStep + xSample) * strataWidth;
            float z = (zStep + zSample) * strataHeight;
            x = x - width / 2.f;
            z = z - height / 2.f;
            mat4 trans = translate(center + vec3(x, 0, z));
            _treeGenerator->generateTree();
            ModuleTree moduleTree = _treeGenerator->getModuleTree();
            addTreeToForest(moduleTree.modules, trans);
        }
    }
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
       vec3 centerPos = vec3(module->getCenterOfMass());
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
bool Forest::checkModuleVoxelOverlap(Module *module, Voxel *voxel, double cellSideLength) {
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


void Forest::artificiallyUpdateTemperatureOfModule(int moduleID, double delta){
    if (_moduleIDs.find(moduleID) != _moduleIDs.end()) {
        Module *m = _moduleIDs[moduleID];
        if (m->getCurrentState()->temperature + delta < maxSimulationTemp) {
            m->getCurrentState()->temperature += delta;
            m->getLastFrameState()->temperature += delta;
        }
    }
}

void Forest::artificiallyUpdateVoxelTemperatureAroundModule(int moduleID, double delta){
    if (_moduleIDs.find(moduleID) != _moduleIDs.end()) {
        VoxelSet voxs = _moduleToVoxels[_moduleIDs[moduleID]];
        for (Voxel *v : voxs){
            if (v->getCurrentState()->temperature + delta < maxSimulationTemp) {
                v->getCurrentState()->temperature += delta;
                v->getLastFrameState()->temperature += delta;
            }
        }
    }
}


/** Init properties of each module based on its branches */
void Forest::initModuleProperties() {
    for (Module *module : _modules) {
        module->initPropertiesFromBranches();
    }
}

void Forest::updateMassAndAreaOfModulesViaBurning(double deltaTimeInMs){
    for (Module *module : _modules) {
        module->updateMassAndAreaViaBurning(deltaTimeInMs, _moduleToVoxels[module]);
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

VoxelSet Forest::getVoxelsMappedToModule(Module *m){
    return _moduleToVoxels[m];
}

ModuleSet Forest::getModulesMappedToVoxel(Voxel *v){
    return _voxelToModules[v];
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

ModuleSet Forest::getModules() {
    return _modules;
}

void Forest::deleteModuleAndChildren(Module *m){
    //First remove it from all the necessary maps
    _moduleIDs.erase(m->ID);
    // Clean up fires
    _fireManager->removeFires(m);

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
    material->cAmbient.r = 0.0f;
    material->cAmbient.g = 0.4f;
    material->cAmbient.b = 0.2f;
    material->cDiffuse.r = 0.0f;
    material->cDiffuse.g = 0.4f;
    material->cDiffuse.b = 0.2f;
    // Create primitive object
    _leaf = std::make_unique<CS123ScenePrimitive>(PrimitiveType::PRIMITIVE_LEAF, *material);
}

/** Initialize the ground primitive */
void Forest::initializeGroundPrimitive() {
    // Initialize ground model
    float axisSize = _voxelGridDim.axisSize;
    mat4 scale = glm::scale(glm::vec3(axisSize, 0, axisSize));
    vec3 gridCenter = _voxelGridDim.center;
    mat4 translate = glm::translate(vec3(gridCenter.x, 0, gridCenter.z));
    _groundModel = translate * scale;
    // Initialize green material for ground
    std::unique_ptr<CS123SceneMaterial> material = std::make_unique<CS123SceneMaterial>();
    material->clear();
    material->cAmbient.r = 0.3f;
    material->cAmbient.g = 0.5f;
    material->cAmbient.b = 0.3f;
    material->cDiffuse.r = 0.1f * 0.73f;
    material->cDiffuse.g = 0.1f * 0.62f;
    material->cDiffuse.b = 0.1f * 0.51f;
    // Create primitive object
    _ground = std::make_unique<CS123ScenePrimitive>(PrimitiveType::PRIMITIVE_GROUND, *material);
}
