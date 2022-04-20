#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"

using namespace std::chrono;


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
};

#endif // SIMULATOR_H
