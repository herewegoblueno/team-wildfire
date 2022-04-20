#include "TreeGenerator.h"
#include "support/Settings.h"
#include "Random.h"
#include "glm/gtx/transform.hpp"
#include <stack>
#include <iostream>
#include <math.h>

TreeGenerator::TreeGenerator() :
    _lSystem(nullptr)
{
    initializeLSystem();
    _leafPreTransform = glm::translate(glm::vec3(0.1, 0, 0)) *
            glm::scale(glm::vec3(0.15));
    // We want our trunk to be thinner than the unit cylinder
    // and have its base at the origin
    glm::mat4 trunkTranslate = glm::translate(glm::vec3(0, 0.5, 0));
    glm::mat4 trunkScale = glm::scale(glm::vec3(trunkInitRadius, 1.0f, trunkInitRadius));
    _trunkPreTransform = trunkTranslate * trunkScale;
}

TreeGenerator::~TreeGenerator() {
    for (Branch *branch : _lifetimeBranches) {
        delete branch;
    }
    for (Module *module : _lifetimeModules) {
        delete module;
    }
}

/**
 *  Generate a new tree. Regenerating the L-system creates potentially
 *  new topology, while random elements of the mesh generation ensure other
 *  variation.
 */
void TreeGenerator::generateTree() {
    // Generate new L-system
    std::string lSystemString = _lSystem->applyRules(recursionDepth);
    // Clear old tree
    _branches.clear();
    _root = nullptr;
    // Convert to mesh
    parseLSystem(lSystemString);
}

/** Parse L-system into branches that contain parent and model matrix info */
void TreeGenerator::parseLSystem(std::string lSystemString) {
    // Stacks for tracking current position and parent branch
    std::stack<glm::mat4> baseCtmStack;
    std::stack<glm::mat4> branchCtmStack;
    std::stack<Branch *> parentStack;
    glm::mat4 baseCtm = glm::mat4(1.0f); // model matrix for base of branch
    glm::mat4 branchCtm = glm::mat4(1.0f); // scaling for branch
    glm::vec3 branchVector = glm::vec3(0, 1, 0); // current dir and length of branch
    Branch *currentParent = nullptr;
    int recursiveDepth = 0;
    for (int i = 0; i < lSystemString.length(); i++) {
        glm::mat4 yRotate; // rotation around y-axis
        glm::mat4 xRotate; // rotation around x-axis
        switch(lSystemString[i]) {
        case '>':
            branchCtm = branchCtm * glm::scale(glm::vec3(branchWidthDecay,
                                             getBranchLength(), branchWidthDecay));
            break;
        case '+':
            yRotate = glm::rotate(getYRotateAnglePlus(), glm::vec3(0, 1, 0));
            xRotate = glm::rotate(getXRotateAngle(), glm::vec3(1, 0, 0));
            baseCtm = baseCtm * yRotate * xRotate;
            branchCtm = branchCtm * yRotate * xRotate;
            break;
        case '-':
            yRotate = glm::rotate(getYRotateAngleMinus(), glm::vec3(0, 1, 0));
            xRotate = glm::rotate(-1.f * getXRotateAngle(), glm::vec3(1, 0, 0));
            baseCtm = baseCtm * yRotate * xRotate;
            branchCtm = branchCtm * yRotate * xRotate;
            break;
        case '[':
            recursiveDepth++;
            baseCtmStack.push(baseCtm);
            branchCtmStack.push(branchCtm);
            parentStack.push(currentParent);
            break;
        case ']':
            recursiveDepth--;
            if (baseCtmStack.empty() || branchCtmStack.empty()) {
                std::cerr << "Error: L-system malformed, attempted to pop from empty stack"
                          << std::endl;

            } else {
                baseCtm = baseCtmStack.top();
                branchCtm = branchCtmStack.top();
                currentParent = parentStack.top();
                baseCtmStack.pop();
                branchCtmStack.pop();
                parentStack.pop();
            }
            break;
        case 'F':
            // Allocate new branch object
            Branch *branch = new Branch;
            _branches.insert(branch);
            _lifetimeBranches.insert(branch);
            if (currentParent == nullptr) {
                branch->radius = trunkInitRadius;
                branch->parent = nullptr;
                branch->model = _trunkPreTransform;
                _root = branch;
            } else {
                branch->radius = currentParent->radius * branchWidthDecay;
                branch->parent = currentParent;
                branch->model = branchCtm * _trunkPreTransform;
                currentParent->children.insert(branch);
            }
            // Add leaves to branch based on leaf density
            bool canAddLeaves = recursiveDepth > minLeafRecursiveDepth
                    && recursiveDepth < maxLeafRecursiveDepth;
            if (canAddLeaves && randomFloat() <= settings.leafDensity) {
                glm::mat4 leafCtm = baseCtm * _leafPreTransform;
                glm::mat4 leafCtmRotated = leafCtm * getYRotateAnglePlus();
                branch->leafModels.push_back(leafCtm);
                branch->leafModels.push_back(leafCtmRotated);
            }
            // Update branch direction and size based on new ctm
            branchVector = glm::vec3(branchCtm * glm::vec4(0, 1, 0, 0));
            // Translate along branch to get to new branching point
            glm::mat4 translate = glm::translate(branchVector);
            baseCtm = translate * baseCtm;
            branchCtm = translate * branchCtm;
            currentParent = branch;
            break;
        }
    }
}

/** Create LSystem, set axiom, and define rewriting rules */
void TreeGenerator::initializeLSystem() {
    _lSystem = std::make_unique<LSystem>();
    _lSystem->setAxiom("FX");
    OutputProbability branchLeftAndRight = OutputProbability(">[-FX]+FX", 0.8f);
    OutputProbability branchLeftOnly = OutputProbability(">[-FX]", 0.2f);
    std::vector<OutputProbability> outputDistribution;
    outputDistribution.push_back(branchLeftAndRight);
    outputDistribution.push_back(branchLeftOnly);
    _lSystem->addRule('X', outputDistribution);
}

/**
 *  Given a tree as a directed graph of branches, split the graph into
 *  a few modules, each containing a portion of the tree
 */
ModuleTree TreeGenerator::branchTreeToModules(BranchTree branchTree) {
    ModuleSet treeModules;
    Branch *rootBranch = branchTree.root;
    Module *rootModule = new Module;
    _lifetimeModules.insert(rootModule);
    treeModules.insert(rootModule);
    ModuleSet newModules = splitIntoModules(rootBranch, rootModule);
    for (Module *module : newModules) {
        treeModules.insert(module);
    }
    for (int i = 0; i < numModuleIterations; i++) {
        ModuleSet updatedNewModules;
        for (Module *module : newModules) {
           ModuleSet toAdd = splitIntoModules(module->rootBranch, module);
           for (Module *moduleToAdd : toAdd) {
               treeModules.insert(moduleToAdd);
               updatedNewModules.insert(moduleToAdd);
           }
        }
        newModules = updatedNewModules;
    }
    std::cout << treeModules.size() << " modules on this tree" << std::endl;
    return ModuleTree(rootModule, treeModules);
}

/**
 * Takes in a root branch and a reference to a root module
 * to use as the branching point for splitting a tree into (usually 3) modules.
 * Return the new modules.
 */
ModuleSet TreeGenerator::splitIntoModules(Branch *rootBranch, Module *rootModule) {
    ModuleSet newModules;
    rootModule->includesRoot = true;
    rootModule->branches.insert(rootBranch);
    for (Branch *branch : rootBranch->children) {
        rootModule->branches.insert(branch);
        Module *module = accumulateModuleFrom(branch);
        module->parent = rootModule;
        module->rootBranch = branch;
        rootModule->children.insert(module);
        newModules.insert(module);
    }
    return newModules;
}

/**
 *  Use DFS to accumulate all branches from root into a module.
 *  Don't include the root branch in the returned module.
 */
Module *TreeGenerator::accumulateModuleFrom(Branch *root) {
    Module *module = new Module;
    _lifetimeModules.insert(module);
    std::stack<Branch *> stack;
    stack.push(root);
    while (!stack.empty()) {
        Branch *v = stack.top();
        stack.pop();
        if (!module->branches.count(v)) {
            module->branches.insert(v);
            for (Branch *child : v->children) {
                stack.push(child);
            }
        }
    }
    module->branches.erase(root);
    return module;
}


/** Return y-axis rotation angle for '+' symbol */
float TreeGenerator::getYRotateAnglePlus() {
    return thetaPlus + randomFloat() * m_pi * settings.branchStochasticity;
}

/** Return y-axis rotation angle for '-' symbol */
float TreeGenerator::getYRotateAngleMinus() {
    return thetaMinus + randomFloat() * m_pi * settings.branchStochasticity;
}

/** Return random angle for x-axis rotation */
float TreeGenerator::getXRotateAngle() {
    return baseXRotation + randomFloat() * 0.3f * settings.branchStochasticity;
}

/** Return relative length of next branch */
float TreeGenerator::getBranchLength() {
    return branchWidthDecay + randomFloat() * 0.1f * settings.branchStochasticity;
}

/** Return the root and branches */
ModuleTree TreeGenerator::getModuleTree() {
    BranchTree branchTree(_root, _branches);
    return branchTreeToModules(branchTree);
}

BranchSet TreeGenerator::getBranches() {
    return _branches;
}
