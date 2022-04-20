#include "simulator.h"
#include <thread>

const int Simulator::NUMBER_OF_SIMULATION_THREADS = 4;

Simulator::Simulator() {}

void Simulator::init(){
    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}

void Simulator::step(VoxelGrid *grid){
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    milliseconds deltaTime = currentTime - timeLastFrame;

    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread){
        threads.emplace_back(&Simulator::stepThreadHandler, this, grid, deltaTime.count(), gridResolution, x, x + jumpPerThread);
    }
    //Wait for all the threads to terminate
    for (auto& th : threads) th.join();
}


void Simulator::stepThreadHandler(VoxelGrid *grid, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                grid->getVoxel(x, y, z)->getCurrentState()->temperature = rand() % 5 *0.4 + 1;
                grid->getVoxel(x, y, z)->getCurrentState()->u = 0.3f*vec3(rand() % 3 - 1.5, rand() % 2, rand() % 2 - 1);
            }
        }
    }
}
