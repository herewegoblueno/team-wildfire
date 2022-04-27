#ifndef MODULE_H
#define MODULE_H
#include <vector>
#include <unordered_set>
#include "glm/glm.hpp"
#include "simulation/physics.h"

const float branchWidthDecay = 0.7; // Amount to scale x, z size of each successive iteration

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
    double temperature; // surface temperature of module
    double radiusRatio = 1;
};

class Module
{
public:
    Module();
    int ID; // lets us identify modules for visual debugging

    Module *_parent = nullptr;
    ModuleSet _children;
    Branch *_rootBranch = nullptr;
    BranchSet _branches;
    // whether root is a branch in the module or just a pointer
    bool _includesRoot;

    glm::dvec3 getCenter() const;
    void initMass();
    void updateMass();
    ModulePhysicalData *getCurrentState();
    ModulePhysicalData *getLastFrameState();

    void updateLastFrameData();

private:
    ModulePhysicalData _currentPhysicalData;
    ModulePhysicalData _lastFramePhysicalData;

    double getBranchMass(Branch *branch) const;
    double getBranchVolume(Branch *branch) const;
};

#endif // MODULE_H
