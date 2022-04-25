#ifndef TREEGENERATOR_H
#define TREEGENERATOR_H

#include "support/lib/CS123SceneData.h"
#include "LSystem.h"
#include "module.h"
#include <unordered_set>

const float m_pi = 3.14159265359;

// How many extra times to repeat the process of splitting
// tree into modules
const int numModuleIterations = 1;

const int recursionDepth = 6;
const double trunkInitLength = 1.0;
const double trunkInitRadius = 0.5;
// Min/max branching levels for adding leaves
const int minLeafRecursiveDepth = 2;
const int maxLeafRecursiveDepth = 9;
// Base amount of y-axis rotation for '+' symbol
const float thetaPlus = 0.5 * m_pi;
// Base amount of y-axis rotation for '-' symbol
const float thetaMinus = 0.3 * m_pi;
// Base amount of x-axis rotation
const float baseXRotation = 0.3;

class TreeGenerator
{
public:
    TreeGenerator();
    void generateTree();
    ModuleTree getModuleTree();
    BranchSet getBranches();
private:
    std::unique_ptr<LSystem> _lSystem;
    void initializeLSystem();
    void parseLSystem(std::string lSystemString);

    glm::mat4 _trunkPreTransform;
    glm::mat4 _leafPreTransform;


    ModuleSet splitIntoModules(Branch *rootBranch, Module *rootModule);
    Module *accumulateModuleFrom(Branch *root);
    ModuleTree branchTreeToModules(BranchTree branchTree);

    BranchSet _branches;
    Branch *_root;

    float getYRotateAnglePlus();
    float getYRotateAngleMinus();
    float getXRotateAngle();
    double getBranchLength();
};

#endif // TREEGENERATOR_H
