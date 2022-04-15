#ifndef VOXEL_H
#define VOXEL_H

#include "voxelgrid.h"

class VoxelGrid;

struct Voxel {
    Voxel(VoxelGrid *grid, int XIndex, int YIndex, int ZIndex, vec3 center);
    Voxel *getVoxelWithIndexOffset(vec3 offset);

    //Set during initialization
    VoxelGrid *grid;
    int XIndex;
    int YIndex;
    int ZIndex;
    vec3 centerInWorldSpace;

    //Set and changed over the course of simulation
    float mass;
    float temperature;
};

#endif // VOXEL_H
