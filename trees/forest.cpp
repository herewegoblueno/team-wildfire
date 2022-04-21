#include "forest.h"
#include "glm/gtx/transform.hpp"

Forest::Forest(int numTrees, float forestWidth, float forestHeight) :
    _treeGenerator(nullptr),
    _moduleNum(0)
{
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
    _treeGenerator = std::make_unique<TreeGenerator>();
    for (int i = 0; i < numTrees; i++) {
        float x = randomFloat() * forestWidth - forestWidth / 2;
        float z = randomFloat() * forestHeight - forestHeight / 2;
        glm::mat4 trans = glm::translate(glm::vec3(x, 0, z));
        _treeGenerator->generateTree();
        ModuleTree moduleTree = _treeGenerator->getModuleTree();
        addPrimitivesFromModules(moduleTree.modules, trans);
    }
}

/**
 * Parse modules into primitives and model matrices, and apply a final transformation
 * to get the tree in the desired position
 */
void Forest::addPrimitivesFromModules(const ModuleSet &modules, glm::mat4 trans) {
    for (Module *module : modules) {
        _moduleNum++;
        for (Branch *branch : module->branches) {
            PrimitiveBundle branchPrimitive(*_trunk, trans * branch->model, _moduleNum);
            _primitives.push_back(branchPrimitive);
            for (glm::mat4 &leafModel : branch->leafModels) {
                PrimitiveBundle leafPrimitive(*_leaf, trans * leafModel);
                _primitives.push_back(leafPrimitive);
            }
        }
    }
}

/**
 * Parse branches into primitives and model matrices, and apply a final transformation
 * to get the tree in the desired position
 */
void Forest::addPrimitivesFromBranches(const BranchSet &branches, glm::mat4 trans) {
    for (Branch *branch : branches) {
        PrimitiveBundle branchPrimitive(*_trunk, trans * branch->model);
        _primitives.push_back(branchPrimitive);
        for (glm::mat4 &leafModel : branch->leafModels) {
            PrimitiveBundle leafPrimitive(*_leaf, trans * leafModel);
            _primitives.push_back(leafPrimitive);
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
