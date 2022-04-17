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
    Branch *root = new Branch;
    root->radius = trunkInitRadius;
    root->model = _trunkPreTransform;
    root->parent = nullptr;
    _branches.insert(root);
    _lifetimeBranches.insert(root);
    _root = root;
    Branch *currentParent = root;
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
            branch->radius = currentParent->radius * branchWidthDecay;
            branch->parent = currentParent;
            branch->model = branchCtm * _trunkPreTransform;
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
Tree TreeGenerator::getTree() {
    return Tree(_root, _branches);
}
