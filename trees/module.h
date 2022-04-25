#ifndef MODULE_H
#define MODULE_H
#include <vector>
#include <unordered_set>
#include "glm/glm.hpp"

const float woodDensity = 1;
// Amount to scale x, z size of each successive iteration
const float branchWidthDecay = 0.7;

class Module;

struct Branch;
typedef std::unordered_set<Branch *> BranchSet;
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

typedef std::unordered_set<Module *> ModuleSet;
struct ModuleTree {
    Module *root;
    ModuleSet modules;
    ModuleTree(Module *root, ModuleSet modules) :
        root(root), modules(modules)
    {}
};

class Module
{
public:
    Module();

    Module *_parent;
    ModuleSet _children;
    Branch *_rootBranch;
    BranchSet _branches;
    // whether root is a branch in the module or just a pointer
    bool _includesRoot;

    glm::dvec3 getCenter() const;

private:
    // total mass of branches, updated during combustion
    double _mass;
    // surface temperature of module
    double _temp;
    void updateRadii();
    void updateMass();

    double getBranchMass(Branch *branch) const;
    double getBranchVolume(Branch *branch) const;

};

#endif // MODULE_H
