#ifndef VOXEL_H
#define VOXEL_H

#include "voxelgrid.h"
#include <glm.hpp>

class VoxelGrid;

using namespace glm;

struct VoxelPhysicalData {
    float mass;
    float temperature;
    vec3 u;  // velocity field
    vec3 tempGradientFromPrevState; // ∇T (here just to make debugging easier if need be)

    // water coefs
    float q_v; // water vapor
    float q_c; // condensed water
//    float q_r; // rain (ignore for now)
};

class Voxel {
public:
    Voxel(VoxelGrid *grid, int XIndex, int YIndex, int ZIndex, vec3 center);
    Voxel *getVoxelWithIndexOffset(vec3 offset);

    //Making these fucntoins since these structs will be being passed through the simulator several times
    //better to enforce that it's just the pointers being passed around
    VoxelPhysicalData *getCurrentState();
    VoxelPhysicalData *getLastFrameState();
    void switchStates();

    //Set during initialization
    VoxelGrid *grid;
    const int XIndex;
    const int YIndex;
    const int ZIndex;
    const vec3 centerInWorldSpace;

    // ∇T
    vec3 getTemperatureGradientFromPreviousFrame();
    // Tamb
    float static getAmbientTemperature(vec3 pos);

private:
    //Set and changed over the course of simulation
    VoxelPhysicalData currentPhysicalState;
    VoxelPhysicalData lastFramePhysicalState;

};

#endif // VOXEL_H
