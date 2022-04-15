#include "MeshGenerator.h"
#include "support/Settings.h"
#include "Random.h"
#include "glm/gtx/transform.hpp"
#include <stack>
#include <iostream>

MeshGenerator::MeshGenerator() :
    m_lSystem(nullptr)
{
    initializeLSystem();
    initializeTrunkPrimitive();
    initializeLeafPrimitive();
}

/**
 *  Generate a new tree. Regenerating the L-system creates potentially
 *  new topology, while random elements of the mesh generation ensure other
 *  variation.
 */
void MeshGenerator::generateTree() {
    // Generate new L-system
    std::string lSystemString = m_lSystem->applyRules(settings.recursionDepth);
    // Clear old tree
    m_primitives.clear();
    m_transformations.clear();
    // Convert to mesh
    parseLSystem(lSystemString);
}

/** Parse L-system into primitives and transformation matrices */
void MeshGenerator::parseLSystem(std::string lSystemString) {
    // Stacks for storing transformation matrices
    std::stack<glm::mat4> baseCtmStack;
    std::stack<glm::mat4> branchCtmStack;
    // Cumulative transformation matrix for current tree part
    glm::mat4 baseCtm = glm::mat4(1.0f);
    // Branches require additional scaling to get progressively smaller
    glm::mat4 branchCtm = glm::mat4(1.0f);
    // Current direction and length of branch
    glm::vec3 branchVector = glm::vec3(0, 1, 0);
    // Track recursive depth for leaves/fruit
    int recursiveDepth = 0;
    for (int i = 0; i < lSystemString.length(); i++) {
        // Rotation around y-axis
        glm::mat4 yRotate;
        // Rotation around x-axis
        glm::mat4 xRotate;
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
            break;
        case ']':
            recursiveDepth--;
            if (baseCtmStack.empty() || branchCtmStack.empty()) {
                std::cerr << "Error: L-system malformed, attempted to pop from empty stack"
                          << std::endl;

            } else {
                baseCtm = baseCtmStack.top();
                baseCtmStack.pop();
                branchCtm = branchCtmStack.top();
                branchCtmStack.pop();
            }
            break;
        case 'F':
            // Add branch to mesh
            addPrimitive(*m_trunk, branchCtm * m_trunkPreTransform);
            // Add leaves to mesh based on leaf density
            bool canAddLeaves = recursiveDepth > minLeafRecursiveDepth
                    && recursiveDepth < maxLeafRecursiveDepth;
            if (canAddLeaves && randomFloat() <= settings.leafDensity) {
                glm::mat4 leafCtm = baseCtm * m_leafPreTransform;
                addPrimitive(*m_leaf, leafCtm);
                glm::mat4 leafCtmRotated = leafCtm * getYRotateAnglePlus();
                addPrimitive(*m_leaf, leafCtmRotated);
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
void MeshGenerator::initializeLSystem() {
    m_lSystem = std::make_unique<LSystem>();
    m_lSystem->setAxiom("FX");
    OutputProbability branchLeftAndRight = OutputProbability(">[-FX]+FX", 0.8f);
    OutputProbability branchLeftOnly = OutputProbability(">[-FX]", 0.2f);
    std::vector<OutputProbability> outputDistribution;
    outputDistribution.push_back(branchLeftAndRight);
    outputDistribution.push_back(branchLeftOnly);
    m_lSystem->addRule('X', outputDistribution);
}

/** Initialize the cylinder building block of our tree trunk/branches */
void MeshGenerator::initializeTrunkPrimitive() {
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
    m_trunk = std::make_unique<CS123ScenePrimitive>(
                PrimitiveType::PRIMITIVE_TRUNK, *material);
    // We want our trunk to be thinner than the unit cylinder
    // and have its base at the origin
    glm::mat4 trunkTranslate = glm::translate(glm::vec3(0, 0.5, 0));
    glm::mat4 trunkScale = glm::scale(glm::vec3(0.5f, 1.0f, 0.5f));
    m_trunkPreTransform = trunkTranslate * trunkScale;
}


/** Initialize the leaf primitive */
void MeshGenerator::initializeLeafPrimitive() {
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
    m_leaf = std::make_unique<CS123ScenePrimitive>(
                PrimitiveType::PRIMITIVE_LEAF, *material);
    m_leafPreTransform = glm::translate(glm::vec3(0.1, 0, 0)) * glm::scale(glm::vec3(0.15));
}

/** Return y-axis rotation angle for '+' symbol */
float MeshGenerator::getYRotateAnglePlus() {
    return thetaPlus + randomFloat() * pi * settings.branchStochasticity;
}

/** Return y-axis rotation angle for '-' symbol */
float MeshGenerator::getYRotateAngleMinus() {
    return thetaMinus + randomFloat() * pi * settings.branchStochasticity;
}

/** Return random angle for x-axis rotation */
float MeshGenerator::getXRotateAngle() {
    return baseXRotation + randomFloat() * 0.3f * settings.branchStochasticity;
}

/** Return relative length of next branch */
float MeshGenerator::getBranchLength() {
    return branchWidthDecay + randomFloat() * 0.1f * settings.branchStochasticity;
}

/** Add a primitive along with its cumulative transformation matrix */
void MeshGenerator::addPrimitive(CS123ScenePrimitive scenePrimitive,
                                 glm::mat4 transformation) {
    m_primitives.push_back(scenePrimitive);
    m_transformations.push_back(transformation);
}

/** Return the vector of primitives */
std::vector<CS123ScenePrimitive> MeshGenerator::getPrimitives() {
    return m_primitives;
}

/** Return the vector of transformation matrices */
std::vector<glm::mat4> MeshGenerator::getTransformations() {
    return m_transformations;
}
