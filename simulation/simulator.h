#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"

using namespace std::chrono;


const float sealevel_temperature = 280;
const float sealevel_pressure = 1;

class Simulator
{
public:
    static const int NUMBER_OF_SIMULATION_THREADS;
    static const float RADIATIVE_COOLING_TERM;
    static const float HEAT_DIFFUSION_INTENSITY_TERM;

    Simulator();
    void init();
    void step(VoxelGrid *grid);
    void cleanupForNextStep(VoxelGrid *grid);


private:
    milliseconds timeLastFrame;
    void stepThreadHandler(VoxelGrid *grid, int deltaTime, int resolution, int minX, int maxX);
    void stepCleanupThreadHandler(VoxelGrid *grid, int resolution, int minX, int maxX);
    void stepVoxelHeatTransfer(Voxel* v, int deltaTimeIn);

    void stepVoxelWater(Voxel* v, int deltaTimeIn);


    static float advect(float field, glm::vec3 vel, glm::vec3 field_grad, float dt);
    static float saturate(float pressure, float temperature);
    static float absolute_temp(float height);
    static float absolute_pres(float height);
    // water particle related equation
};

#endif // SIMULATOR_H
