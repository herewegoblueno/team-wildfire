#include "module.h"
#include <iostream>

#include <chrono>
using namespace std::chrono;

using namespace glm;

Module::Module() :
    ID(rand() % 1000000),
    _warning(false),
    _includesRoot(false)
{
    // TODO: init temperature to ambient temp
}

/** Init mass of module based on branch masses */
void Module::initMass() {
    _currentPhysicalData.mass = 0;
    _lastFramePhysicalData.mass = 0;
    for (Branch *branch : _branches) {
        _currentPhysicalData.mass += getBranchMass(branch);
        _lastFramePhysicalData.mass += getBranchMass(branch);
    }
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
        dvec3 branchBase = dvec3(branch->model*trunkObjectBottom);
        dvec3 branchEnd = dvec3(branch->model*trunkObjectTop);
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

ModulePhysicalData *Module::getLastFrameState() {
    return &_lastFramePhysicalData;
}

double Module::getBranchMass(Branch *branch) const {
    return getBranchVolume(branch) * woodDensity;
}

double Module::getBranchVolume(Branch *branch) const {
    double l = branch->length;
    double r0 = branch->radius * _lastFramePhysicalData.radiusRatio;
    double r1 = r0 * branchWidthDecay;
    return (M_PI / 3.0)* l * (r0*r0 + r0*r1 + r1*r1);
}

void Module::updateLastFrameData(){
    _currentPhysicalData.radiusRatio = abs(std::sin(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() / 30000.0));
    _lastFramePhysicalData = _currentPhysicalData;
}
