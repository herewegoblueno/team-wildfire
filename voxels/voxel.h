#ifndef VOXEL_H
#define VOXEL_H

#include "voxelgrid.h"
#include <glm.hpp>

class VoxelGrid;

using namespace glm;

struct VoxelPhysicalData {
    double mass = 0;
    double temperature = 0;
    dvec3 tempGradientFromPrevState = dvec3(0,0,0); // ∇T (here just to make debugging easier if need be)
    double tempLaplaceFromPrevState = 0; // ∇^2T (here just to make debugging easier if need be)

    dvec3 u = dvec3(0,0,0);  // velocity field

    // water coefs
    float q_v = 0; // water vapor
    float q_c = 0; // condensed water
    float q_r = 0; // rain (ignore for now)
};

struct VoxelTemperatureGradientInfo {
    dvec3 gradient; // ∇T
    double laplace; // ∇^2T
};

class Voxel {
public:
    Voxel(VoxelGrid *grid, int XIndex, int YIndex, int ZIndex, vec3 center);
    Voxel *getVoxelWithIndexOffset(vec3 offset);

    //Making these fucntoins since these structs will be being passed through the simulator several times
    //better to enforce that it's just the pointers being passed around
    VoxelPhysicalData *getCurrentState();
    VoxelPhysicalData *getLastFrameState();
    void updateLastFrameData();

    //Set during initialization
    VoxelGrid *grid;
    const int XIndex;
    const int YIndex;
    const int ZIndex;
    const vec3 centerInWorldSpace;

    VoxelTemperatureGradientInfo getTemperatureGradientInfoFromPreviousFrame();
    double static getAmbientTemperature(vec3 pos);

    vec3 getGradient(float (*func)(Voxel *));

private:
    //Set and changed over the course of simulation
    VoxelPhysicalData currentPhysicalState;
    VoxelPhysicalData lastFramePhysicalState;

    double getNeighbourTemperature(int xOffset, int yOffset, int zOffset);
};

#endif // VOXEL_H
