#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"
#include "physics.h"
#include "fluid.h"
#include "trees/forest.h"
#include <Eigen/Sparse>

using namespace std::chrono;



class Simulator
{
public:
    static const int NUMBER_OF_SIMULATION_THREADS;

    Simulator();
    void init();
    void step(VoxelGrid *grid, Forest *forest = nullptr);
    void linear_step(VoxelGrid *grid, Forest *forest = nullptr);
    void cleanupForNextStep(VoxelGrid *grid, Forest *forest = nullptr);


private:
    milliseconds timeLastFrame;
    void stepThreadHandler(VoxelGrid *grid, Forest *forest, int deltaTime, int resolution, int minX, int maxX);
    void stepThreadHandlerWind(VoxelGrid *grid, Forest *forest, double deltaTime, int resolution, int minX, int maxX,
                               double* diag, double* rhs, int* id_xyz);
    void stepThreadHandlerWater(VoxelGrid *grid, Forest *forest, double deltaTime, int resolution, int minX, int maxX,
                                double* pressure);
    void stepCleanupThreadHandler(VoxelGrid *grid, Forest *forest, int resolution, int minX, int maxX);

    void stepVoxelHeatTransfer(Voxel* v, ModuleSet nearbyModules, int deltaTimeInMs);

    void stepVoxelWater(Voxel* v, double deltaTimeInMs);
    void stepVoxelWind(Voxel* v, double deltaTimeInMs);

    void stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs);


};

#endif // SIMULATOR_H
