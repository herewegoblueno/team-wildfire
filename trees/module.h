#ifndef MODULE_H
#define MODULE_H
#include <vector>
#include <unordered_set>
#include "glm/glm.hpp"
#include "simulation/physics.h"

const float branchWidthDecay = 0.7; // Amount to scale x, z size of each successive iteration

// Object space values based off of Trunk.cpp
const glm::vec4 trunkObjectBottom = glm::vec4(0, -0.5, 0, 1);
const glm::vec4 trunkObjectTop = glm::vec4(0, 0.5, 0, 1);
const double trunkObjectRadius = 0.5;
const double trunkObjectLength = 1.0;

enum ModuleVisualizationModes { ID, MODULE_TEMPERATURE };

class Module;
struct Branch;
typedef std::unordered_set<Branch *> BranchSet;
typedef std::unordered_set<Module *> ModuleSet;

struct Branch {
    Branch *parent;
    BranchSet children;
    glm::mat4 model; // gives branch world-space position
    glm::mat4 invModel; // inverse of model
    double length;
    double radius; // radius of base
    std::vector<glm::mat4> leafModels; // gives leaves world-space position
    int moduleID; // lets us identify modules for visual debugging
};

struct BranchTree {
    Branch *root;
    BranchSet branches;
    BranchTree(Branch *root, BranchSet branches) :
        root(root), branches(branches)
    {}
};

struct ModuleTree {
    Module *root;
    ModuleSet modules;
    ModuleTree(Module *root, ModuleSet modules) :
        root(root), modules(modules)
    {}
};

struct ModulePhysicalData {
    double mass;        // total mass of branches, updated during combustion
    double area;        // total surface area of all the branches
    double temperature; // surface temperature of module
    double radiusRatio = 1;
};

class Module
{
public:
    Module();
    int ID; // lets us identify modules for visual debugging
    bool _warning; // if true, color red for visual debugging

    Module *_parent = nullptr;
    ModuleSet _children;
    Branch *_rootBranch = nullptr;
    BranchSet _branches;
    // whether root is a branch in the module or just a pointer
    bool _includesRoot;

    glm::dvec3 getCenter() const;
    void initMassAndArea();
    void updateMassAndAreaViaBurning(double deltaTimeInMs);
    ModulePhysicalData *getCurrentState();
    ModulePhysicalData *getLastFrameState();

    double getMassLossRateFromPreviousFrame();
    double getTemperatureLaplaceFromPreviousFrame();
    void updateLastFrameData();

private:
    ModulePhysicalData _currentPhysicalData;
    ModulePhysicalData _lastFramePhysicalData;

    double getBranchMass(Branch *branch) const;
    double getBranchVolume(Branch *branch) const;
    double getBranchLateralSurfaceArea(Branch *branch) const;

    double getReactionRateFromPreviousFrame();
    double sigmoidFunc(double x);
    double getMassLossDueToBurning(double deltaTimeInMs);
    void updateRadiiToReflectMassLoss(double massLoss);
};

#endif // MODULE_H
