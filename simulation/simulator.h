#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"
#include "physics.h"
#include "fluid.h"
#include "trees/forest.h"

using namespace std::chrono;

struct host2cuda_data
{
    double* grid_temp;
    double* grid_q_v;
    double* grid_q_c;
    double* grid_q_r;
    double* grid_h;
    double* grid_humidity;
    double* u_xyz;
    int* id_xyz;
};

class Simulator
{
public:
    static const int NUMBER_OF_SIMULATION_THREADS;
    static const int MAX_TIMESTEP_MS_ALLOWED;

    Simulator();
    void init();
    void step(VoxelGrid *grid, Forest *forest = nullptr);
    void cleanupForNextStep(VoxelGrid *grid, Forest *forest = nullptr);
    float getTimeSinceLastFrame();

private:
    milliseconds timeLastFrame;
    float timeSinceLastFrame;
    void stepThreadHeatHandler(VoxelGrid *grid, Forest *forest, int deltaTime, int resolution, int minX, int maxX);
    void stepThreadWaterHandler(VoxelGrid *grid, int deltaTime, int resolution, int minXInclusive, int maxXExclusive);
    void stepCleanupThreadHandler(VoxelGrid *grid, Forest *forest, int resolution, int minX, int maxX);

    void stepVoxelHeatTransfer(Voxel* v, ModuleSet nearbyModules, int deltaTimeInMs);

    void stepVoxelWater(Voxel* v, double deltaTimeInMs);
    void stepVoxelWind(Voxel* v, double deltaTimeInMs);

    void stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs);
    host2cuda_data host2cuda;
    void mallocHost2cuda(VoxelGrid *grid);
    void writeHost2Cuda(Voxel* v, int index);
    void writeCuda2Host(Voxel* v, int index);
    void freeHost2cuda();

};

#endif // SIMULATOR_H
