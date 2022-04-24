#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <chrono>
#include "voxels/voxelgrid.h"
#include "physics.h"

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

    void stepVoxelHeatTransfer(Voxel* v, int deltaTimeInMs);
    void stepVoxelWater(Voxel* v, int deltaTimeInMs);


    // water particle related equation
    static float advect(float field, glm::vec3 vel, glm::vec3 field_grad, float dt);
    static float saturate(float pressure, float temperature);
    static float absolute_temp(float height);
    static float absolute_pres(float height);
    static float mole_fraction(float ratio);
    static float avg_mole_mass(float ratio);
    static float isentropic_exponent(float ratio);
    static float heat_capacity(float gamma, float mass);

};

#endif // SIMULATOR_H
