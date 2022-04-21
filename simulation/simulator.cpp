#include "simulator.h"
#include <thread>
#include "support/Settings.h"

const int Simulator::NUMBER_OF_SIMULATION_THREADS = 4;

Simulator::Simulator() {}

void Simulator::init(){
    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}

void Simulator::step(VoxelGrid *grid){
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    int deltaTime = (currentTime - timeLastFrame).count();
    deltaTime *= settings.simulatorTimescale;

    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandler, this, grid, deltaTime, gridResolution, x, x + jumpPerThread);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate
}

void Simulator::cleanupForNextStep(VoxelGrid *grid){
    //No need for asserts here, same asserts as in step()
    int gridResolution = grid->getResolution();
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepCleanupThreadHandler, this, grid, gridResolution, x, x + jumpPerThread);

    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}


void Simulator::stepThreadHandler(VoxelGrid *grid, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                stepVoxelHeatTransfer(grid->getVoxel(x, y, z), deltaTime);
            }
        }
    }
}

void Simulator::stepCleanupThreadHandler(VoxelGrid *grid, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                grid->getVoxel(x, y, z)->switchStates();
            }
        }
    }
}

void Simulator::stepVoxelWater(Voxel* v, int deltaTimeIn)
{
    float q_v = v->getCurrentState()->q_v;
    float q_c = v->getCurrentState()->q_c;
    glm::vec3 u = v->getCurrentState()->u;
    float h = v->centerInWorldSpace.y; // need discussion

    // TODO: gradients
//    q_v = advect(q_v, u, q_v_grad, deltaTimeIn);
//    q_c = advect(q_c, u, q_c_grad, deltaTimeIn);
    float q_vs = saturate(absolute_pres(h), absolute_temp(h));

    float E_r;// TODO
    float A_c, K_c; // TODO
    q_v = q_v + std::min(q_vs - q_v, q_c) + E_r;
    q_c = q_c - std::min(q_vs - q_v, q_c) - A_c - K_c;


}



// Advection function based on *Stable Fluids* Jos Stam 1999
float Simulator::advect(float field, glm::vec3 vel, glm::vec3 field_grad, float dt)
{
    float df_dt = -glm::dot(vel, field_grad);
    return field + df_dt*dt;
}

// saturation ratio calculation of Eq.16 Stormscape
float Simulator::saturate(float pressure, float temperature)
{
    return 380.16/pressure*exp(17.67*temperature/(temperature+243.5));
}


// absolute temperature calculation based on Eq.27 Stormscape
float Simulator::absolute_temp(float height)
{
    return sealevel_temperature - 0.0065*height;
}

// absolute pressure calculation based on Eq.28 Stormscape
float Simulator::absolute_pres(float height)
{
    float tmp = 1 - 0.0065*height/sealevel_temperature;
    return sealevel_pressure*pow(tmp, 5.2561);
}
