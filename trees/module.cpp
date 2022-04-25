#include "module.h"

Module::Module() :
    _includesRoot(false)
{

}

/** Return center of mass of module */
glm::dvec3 Module::getCenter() const {
    glm::dvec3 center(0);
    for (Branch *branch : _branches) {
        glm::dvec3 branchBase = glm::dvec3(branch->model*glm::vec4(0));
        glm::dvec3 branchDir = glm::dvec3(branch->model*glm::vec4(0, 1, 0, 0));
        glm::dvec3 branchCenter = branchBase +
                glm::normalize(branchDir) * branch->length / 2.0;
        double mass = getBranchMass(branch);
        center += mass * branchCenter;
    }
    return center / _mass;
}

double Module::getBranchMass(Branch *branch) const {
    return getBranchVolume(branch) * woodDensity;
}

double Module::getBranchVolume(Branch *branch) const {
    double l = branch->length;
    double r0 = branch->radius;
    double r1 = r0 * branchWidthDecay;
    return (M_PI / 3.0)* l * (r0*r0 + r0*r1 + r1*r1);
}
