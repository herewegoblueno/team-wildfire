#include "simulator.h"

Simulator::Simulator()
{

}

void Simulator::init(){
    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}

void Simulator::step(VoxelGrid *grid){
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    milliseconds deltaTime = currentTime - timeLastFrame;

    int gridResolution = grid->getResolution();

    for (int x = 0; x < gridResolution; x++){
        for (int y = 0; y < gridResolution; y++){
            for (int z = 0; z < gridResolution; z++){
                grid->getVoxel(x, y, z)->getCurrentState()->temperature = rand() % 5;
                grid->getVoxel(x, y, z)->getCurrentState()->u = vec3(rand() % 5 - 2.5, rand() % 1, rand() % 5 - 2.5);
            }
        }
    }
}

