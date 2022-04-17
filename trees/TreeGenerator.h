#ifndef TREEGENERATOR_H
#define TREEGENERATOR_H

#include "support/lib/CS123SceneData.h"
#include "LSystem.h"
#include "module.h"
#include <unordered_set>

struct Tree {
    Branch *root;
    BranchSet branches;
    Tree(Branch *root, BranchSet branches) :
        root(root), branches(branches)
    {}
};

const float m_pi = 3.14159265359;

const int recursionDepth = 6;
const float trunkInitRadius = 0.5;
// Min/max branching levels for adding leaves
const int minLeafRecursiveDepth = 2;
const int maxLeafRecursiveDepth = 9;
// Base amount of y-axis rotation for '+' symbol
const float thetaPlus = 0.5 * m_pi;
// Base amount of y-axis rotation for '-' symbol
const float thetaMinus = 0.3 * m_pi;
// Base amount of x-axis rotation
const float baseXRotation = 0.3;
// Amount to scale x, z size of each successive iteration
const float branchWidthDecay = 0.7;

class TreeGenerator
{
public:
    TreeGenerator();
    ~TreeGenerator();
    void generateTree();
    Tree getTree();
private:
    std::unique_ptr<LSystem> _lSystem;
    void initializeLSystem();
    void parseLSystem(std::string lSystemString);

    glm::mat4 _trunkPreTransform;
    glm::mat4 _leafPreTransform;

    BranchSet _branches;
    Branch *_root;
    BranchSet _lifetimeBranches; // for memory management

    float getYRotateAnglePlus();
    float getYRotateAngleMinus();
    float getXRotateAngle();
    float getBranchLength();
};

#endif // TREEGENERATOR_H
