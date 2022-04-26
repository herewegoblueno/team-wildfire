#include "module.h"
#include <iostream>

using namespace glm;

Module::Module() :
    ID(rand() % 1000000),
    _includesRoot(false)
{
    // TODO: init temperature to ambient temp
}

/** Update mass of module based on branch masses */
void Module::updateMass() {
    _currentPhysicalData.mass = 0;
    for (Branch *branch : _branches) {
        _currentPhysicalData.mass += getBranchMass(branch);
    }
}

/** Return center of mass of module */
dvec3 Module::getCenter() const {
    dvec3 center(0);
    for (Branch *branch : _branches) {
        dvec3 branchBase = dvec3(branch->model*vec4(0));
        dvec3 branchEnd = dvec3(branch->model*vec4(0, 1, 0, 1));
        dvec3 branchDir = branchEnd - branchBase;
        dvec3 branchCenter = branchBase +
                normalize(branchDir) * branch->length / 2.0;
        double mass = getBranchMass(branch);
        center += mass * branchCenter;
    }
    return center / _currentPhysicalData.mass;
}

ModulePhysicalData *Module::getCurrentState() {
    return &_currentPhysicalData;
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
