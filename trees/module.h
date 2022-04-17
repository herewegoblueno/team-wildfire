#ifndef MODULE_H
#define MODULE_H
#include <vector>
#include <unordered_set>
#include "glm/glm.hpp"

struct Branch {
    Branch *parent;
    //std::vector<Branch *> children;
    glm::mat4 model; // gives branch world-space position
    double radius; // radius of base
    std::vector<glm::mat4> leafModels; // gives leaves world-space position
};

typedef std::unordered_set<Branch *> BranchSet;

class Module
{
public:
    Module();


private:
    std::vector<Branch> _branches;
    // total mass of branches, updated during combustion
    double _mass;
    // surface temperature of module
    double _temp;
    void updateRadii();
    void updateMass();

};

#endif // MODULE_H
