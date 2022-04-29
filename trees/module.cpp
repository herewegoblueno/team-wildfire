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
}

/** Init mass and surface area of module based on branches  */
void Module::initMassAndArea() {
    _currentPhysicalData.mass = 0;
    _lastFramePhysicalData.mass = 0;
    _currentPhysicalData.area = 0;
    _lastFramePhysicalData.area = 0;
    for (Branch *branch : _branches) {
        double mass = getBranchMass(branch);
        double area = getBranchLateralSurfaceArea(branch);
        _currentPhysicalData.mass += mass;
        _lastFramePhysicalData.mass += mass;
        _currentPhysicalData.area += area;
        _lastFramePhysicalData.area += area ;
    }
}

/** Update mass and surface area of module based on branches */
void Module::updateMassAndArea() {
    _currentPhysicalData.mass = 0;
    _currentPhysicalData.area = 0;
    for (Branch *branch : _branches) {
        _currentPhysicalData.mass += getBranchMass(branch);
        _currentPhysicalData.area += getBranchLateralSurfaceArea(branch) ;
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

//getBranchLateralSurfaceArea and getBranchVolume are called by initMassAndArea and updateMassAndArea,
//functions used for both module initialization and updating, so it's better to use _currentPhysicalData
double Module::getBranchLateralSurfaceArea(Branch *branch) const {
    double l = branch->length;
    double r0 = branch->radius * _currentPhysicalData.radiusRatio;
    double r1 = r0 * branchWidthDecay;
    return (M_PI) * (r0 + r1) * std::sqrt(pow(r0 - r1, 2) + pow(l, 2));
}

double Module::getBranchVolume(Branch *branch) const {
    double l = branch->length;
    double r0 = branch->radius * _currentPhysicalData.radiusRatio;
    double r1 = r0 * branchWidthDecay;
    return (M_PI / 3.0)* l * (r0*r0 + r0*r1 + r1*r1);
}

/**
 * Compute Laplace based on temperature gradients between this<->parent and this<->children.
 * If this module is missing children or parent, we use this module's temperature as a
 * replacement value, essentially padding the module tree with another equi-temperature module.
 */
double Module::getTemperatureLaplaceFromPreviousFrame() {
    double thisTemp = getLastFrameState()->temperature;
    dvec3 thisCenter = getCenter();
    // averages over all children
    double distToChildren;
    double childrenTemp;
    if (_children.size() > 0) {
        dvec3 childrenCenter = dvec3(0.0);
        childrenTemp = 0.0;
        for (Module *child : _children) {
            childrenTemp += child->getLastFrameState()->temperature;
            childrenCenter += child->getCenter();
        }
        double childrenSize = static_cast<double>(_children.size());
        childrenTemp = childrenTemp / childrenSize;
        childrenCenter = childrenCenter / childrenSize;
        distToChildren = length(childrenCenter - thisCenter);
    } else {
        childrenTemp = thisTemp;
        distToChildren = 0;
    }

    double parentTemp;
    double distToParent;
    if (_parent != nullptr) {
        parentTemp = _parent->getLastFrameState()->temperature;
        distToParent = length(thisCenter - _parent->getCenter());
    } else {
        parentTemp = thisTemp;
        distToParent = 0;
    }

    // if missing parent or children, approximate total distance
    // by using the non-missing value twice
    if (!distToParent) {
        distToParent = distToChildren;
    } else if (!distToChildren) {
        distToChildren = distToParent;
    }

    double childrenDeriv = (childrenTemp - thisTemp) / distToChildren;
    double parentDeriv = (thisTemp - parentTemp) / distToParent;
    return (childrenDeriv - parentDeriv) / (distToChildren + distToParent);
}

/** Reaction rate based on module temperature */
double Module::getReactionRateFromPreviousFrame() {
    double moduleTemp = getLastFrameState()->temperature;
    if (moduleTemp < reaction_rate_t0) return 0.0;
    if (moduleTemp > reaction_rate_t1) return 1.0;
    double x = (moduleTemp - reaction_rate_t0) / (reaction_rate_t1 - reaction_rate_t0);
    return sigmoidFunc(x);
}

/** Sigmoid-like function for interpolation in reaction rate calculations */
double Module::sigmoidFunc(double x) {
    return 3*std::pow(x, 2) - 2*std::pow(x, 3);
}

/**
 * Mass loss rate (dM/dt) based on Eq. 1 of Fire in Paradise paper
 * TODO (optional): add char insulation and pyrolyzing front area terms
 */
double Module::getMassLossRateFromPreviousFrame() {
    double reactionRate = getReactionRateFromPreviousFrame();
    double moduleTemp = getLastFrameState()->temperature;
    return -reactionRate * moduleTemp;
}

void Module::updateLastFrameData(){
    _lastFramePhysicalData = _currentPhysicalData;
}
