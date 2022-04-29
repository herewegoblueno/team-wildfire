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

    //Making these functions since these structs will be being passed through the simulator several times
    //better to enforce that it's just the pointers being passed around
    VoxelPhysicalData *getCurrentState();
    VoxelPhysicalData *getLastFrameState();
    void updateLastFrameData();

    //Set during initialization
    VoxelGrid *grid;
    const int XIndex;
    const int YIndex;
    const int ZIndex;
    const dvec3 centerInWorldSpace;

    VoxelTemperatureGradientInfo getTemperatureGradientInfoFromPreviousFrame();
    double getAmbientTemperature();

    dvec3 getGradient(double (*func)(Voxel *));
    double getLaplace(double (*func)(Voxel *));
    dvec3 getVerticity();

private:
    //Set and changed over the course of simulation
    VoxelPhysicalData currentPhysicalState;
    VoxelPhysicalData lastFramePhysicalState;

    double getAmbientTempAtIndices(int x, int y, int z);
    double getNeighbourTemperature(int xOffset, int yOffset, int zOffset);

};


double get_q_ux(Voxel* v);
double get_q_uy(Voxel* v);
double get_q_uz(Voxel* v);

double get_ux(Voxel* v);
double get_uy(Voxel* v);
double get_uz(Voxel* v);

double get_q_v(Voxel* v);
double get_q_c(Voxel* v);
double get_q_r(Voxel* v);

#endif // VOXEL_H
