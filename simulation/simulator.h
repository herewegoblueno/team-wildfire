#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"

using namespace std::chrono;


class Simulator
{
public:
    static const int NUMBER_OF_SIMULATION_THREADS;
    Simulator();
    void init();
    void step(VoxelGrid *grid);
    void stepThreadHandler(VoxelGrid *grid, int deltaTime, int resolution, int minX, int maxX);


private:
    milliseconds timeLastFrame;
};

#endif // SIMULATOR_H
