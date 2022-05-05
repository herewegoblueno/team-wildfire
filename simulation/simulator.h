#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"
#include "physics.h"
#include "fluid.h"
#include "trees/forest.h"
#include <Eigen/Sparse>

using namespace std::chrono;

struct host2cuda_data
{
    double* grid_temp;
    double* grid_q_v;
    double* grid_h;
    double* u_xyz;
    int* id_xyz;
};

class Simulator
{
public:
    static const int NUMBER_OF_SIMULATION_THREADS;

    Simulator();
    void init();
    void step(VoxelGrid *grid, Forest *forest = nullptr);
    void cleanupForNextStep(VoxelGrid *grid, Forest *forest = nullptr);


private:
    milliseconds timeLastFrame;
    void stepThreadHandler(VoxelGrid *grid, Forest *forest, int deltaTime, int resolution, int minX, int maxX);
    void stepCuda2hostThreadHandler(VoxelGrid *grid ,Forest * forest, int deltaTime, int resolution,
                                                                       int minXInclusive, int maxXExclusive);
    void stepCleanupThreadHandler(VoxelGrid *grid, Forest *forest, int resolution, int minX, int maxX);

    void stepVoxelHeatTransfer(Voxel* v, ModuleSet nearbyModules, int deltaTimeInMs);

    void stepVoxelWater(Voxel* v, double deltaTimeInMs);
    void stepVoxelWind(Voxel* v, double deltaTimeInMs);

    void stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs);
    host2cuda_data host2cuda;
    void mallocHost2cuda(VoxelGrid *grid);
    void writeHost2cudaSpace(Voxel* v, int index);
    void freeHost2cuda();

};

#endif // SIMULATOR_H
