#ifndef VOXEL_H
#define VOXEL_H

#include "voxelgrid.h"
#include <glm.hpp>

class VoxelGrid;

using namespace glm;

struct VoxelPhysicalData {
    float mass;
    float temperature;
    vec3 u;  // velocity
};

class Voxel {
public:
    Voxel(VoxelGrid *grid, int XIndex, int YIndex, int ZIndex, vec3 center);
    Voxel *getVoxelWithIndexOffset(vec3 offset);

    //Making these fucntoins since these structs will be being passed through the simulator several times
    //better to enforce that it's just the pointers being passed around
    VoxelPhysicalData *getCurrentState();
    VoxelPhysicalData *getLastFrameState();

    //Set during initialization
    VoxelGrid *grid;
    int XIndex;
    int YIndex;
    int ZIndex;
    vec3 centerInWorldSpace;

private:
    //Set and changed over the course of simulation
    VoxelPhysicalData currentPhysicalState;
    VoxelPhysicalData lastFramePhysicalState;
};

#endif // VOXEL_H
