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
    initCenterOfMass();
}

void Module::initCenterOfMass(){
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
    _centerOfMass = center / _currentPhysicalData.mass;
}

/** Update mass and surface area of module based on branches (will only change due to burning) */
void Module::updateMassAndAreaViaBurning(double deltaTimeInMs, VoxelSet &voxels) {
    _currentPhysicalData.mass = 0;
    _currentPhysicalData.area = 0;
    _currentPhysicalData.radiusRatio = _lastFramePhysicalData.radiusRatio;
    double massChange = getMassChangeDueToBurning(deltaTimeInMs, voxels);
    updateRadiiToReflectMassLoss(massChange);
    for (Branch *branch : _branches) {
        _currentPhysicalData.mass += getBranchMass(branch);
        _currentPhysicalData.area += getBranchLateralSurfaceArea(branch) ;
    }
}

double Module::getMassChangeDueToBurning(double deltaTimeInMs, VoxelSet &voxels){
   double windSpeed = 0;
   for (Voxel *v : voxels) windSpeed += length(v->getLastFrameState()->u);
   double dMdt = getMassChangeRateFromPreviousFrame(windSpeed);
   _currentPhysicalData.massChangeRateFromLastFrame = dMdt;
   return dMdt * deltaTimeInMs / 1000.0;
}

//Using something similar to raymarching: reduce the radii bit my bit till you get a good mass below target
void Module::updateRadiiToReflectMassLoss(double massChange){
    //Assumes actual mass change can only negative
    if (massChange >= 0 || _lastFramePhysicalData.radiusRatio <= 0) return;
    double targetMass = std::max(0.0, _lastFramePhysicalData.mass + massChange);
    double testMass;
    do {
        testMass = 0;
        _currentPhysicalData.radiusRatio -= 0.0001;
        if (_currentPhysicalData.radiusRatio < 0) _currentPhysicalData.radiusRatio = 0;
        for (Branch *branch : _branches) testMass += getBranchMass(branch);
    }
    while (testMass > targetMass);
}

dvec3 Module::getCenterOfMass() const {
    return _centerOfMass;
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
//Also important since we use it in updateRadiiToReflectMassLoss
double Module::getBranchLateralSurfaceArea(Branch *branch) const {
    double l = branch->length;
    double r0 = branch->radius * _currentPhysicalData.radiusRatio;
    double r1 = r0 * branchWidthDecay;
    return (M_PI) * (r0 + r1) * std::sqrt(std::pow(r0 - r1, 2) + std::pow(l, 2));
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
    dvec3 thisCenter = getCenterOfMass();

    bool hasChildren = _children.size() > 0;
    bool hasParent = _parent != nullptr;

    if (!hasChildren && !hasParent) return 0;

    // averages over all children
    double distToChildren;
    double childrenTemp;
    if (hasChildren) {
        dvec3 childrenCenter = dvec3(0.0);
        childrenTemp = 0.0;
        for (Module *child : _children) {
            childrenTemp += child->getLastFrameState()->temperature;
            childrenCenter += child->getCenterOfMass();
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
    if (hasParent) {
        parentTemp = _parent->getLastFrameState()->temperature;
        distToParent = length(thisCenter - _parent->getCenterOfMass());
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
double Module::getReactionRateFromPreviousFrame(double windSpeed) {
    double moduleTemp = getLastFrameState()->temperature;
    if (moduleTemp < reaction_rate_t0) return 0.0;
    if (moduleTemp > reaction_rate_t1) return 1.0;
    double tempRatio = (moduleTemp - reaction_rate_t0) / (reaction_rate_t1 - reaction_rate_t0);
    double rate = sigmoidFunc(tempRatio);

    //TODO: double check this
    windSpeed = std::min(windSpeed, speed_for_max_wind_boost);
    double windBoost = (max_wind_combustion_boost - 1.0)
            * sigmoidFunc(windSpeed / speed_for_max_wind_boost) + 1.0;
    return windBoost * rate;
}

/**
 * Sigmoid-like function for interpolation between 0 and 1
 * in reaction rate calculations
 */
double Module::sigmoidFunc(double x) {
    return 3*std::pow(x, 2) - 2*std::pow(x, 3);
}

/**
 * Mass loss rate (dM/dt) based on Eq. 1 of Fire in Paradise paper
 * TODO (optional): add char insulation
 */
double Module::getMassChangeRateFromPreviousFrame(double windSpeed) {
    double reactionRate = getReactionRateFromPreviousFrame(windSpeed);
    double surfaceArea = getLastFrameState()->area;
    return -reactionRate * reation_rate_multiplier * surfaceArea;
}

void Module::updateLastFrameData(){
    _lastFramePhysicalData = _currentPhysicalData;
}
