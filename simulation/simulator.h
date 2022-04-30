#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"
#include "physics.h"
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
    void stepCleanupThreadHandler(VoxelGrid *grid, Forest *forest, int resolution, int minX, int maxX);

    void stepVoxelHeatTransfer(Voxel* v, int deltaTimeInMs);
    void stepVoxelWater(Voxel* v, double deltaTimeInMs);
    void stepVoxelWind(Voxel* v, double deltaTimeInMs);

    void stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs);

    // water particle related equation
    static double advect(double (*func)(Voxel *), glm::dvec3 vel, double dt, Voxel* v);
    static double saturate(double pressure, double temperature);
    static double absolute_temp(double height);
    static double absolute_pres(double height);
    static double mole_fraction(double ratio);
    static double avg_mole_mass(double ratio);
    static double isentropic_exponent(double ratio);
    static double heat_capacity(double gamma, double mass);

    // wind related equation
    static glm::dvec3 verticity_confinement(glm::dvec3 u, Voxel* v, double time);
    static void pressure_projection_LLT(VoxelGrid *grid, double time);
    static void pressure_projection_PCG(VoxelGrid *grid, double time);
    static void pressure_projection_Jacobi(VoxelGrid *grid, double time);

};

#endif // SIMULATOR_H
