#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"

using namespace std::chrono;

class Simulator
{
public:
    Simulator();
    void init();
    void step(VoxelGrid *grid);

private:
    milliseconds timeLastFrame;
};

#endif // SIMULATOR_H
