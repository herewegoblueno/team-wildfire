#include "simulator.h"
#include <thread>
#include "support/Settings.h"

const int Simulator::NUMBER_OF_SIMULATION_THREADS = 4;

Simulator::Simulator() {}

void Simulator::init(){
    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}

void Simulator::step(VoxelGrid *grid, Forest *forest){
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    int deltaTime = (currentTime - timeLastFrame).count();
    if (deltaTime > 100) deltaTime = 100;
    deltaTime *= settings.simulatorTimescale;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    if (forest != nullptr){ //Forest is optional
        forest->updateMassAndAreaOfModules();
        for (Module *m : forest->getModules()) {
            VoxelSet surroundingAir = forest->getVoxelsMappedToModule(m);
            stepModuleHeatTransfer(m, surroundingAir, deltaTime);
        }
        //<TODO: Temperature changes, burning, and Radii updates should be here>
        //<TODO: water content of modules should go here>
        forest->updateMassOfVoxels();
    }

    //(lines 7-12 in Algorithm 1 of paper)
    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandler, this, grid, forest, deltaTime, gridResolution, x, x + jumpPerThread);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    if (forest != nullptr){ //Forest is optional
        //This should be the last step of the simulation
        //We want the actual step to be based on last frame's mapping, but we want the scene's rendering
        //(which starts after the step function) to be based on the resultant mapping after the step
        //(so the users see the most up to date state)
        forest->updateModuleVoxelMapping();
    }
}

void Simulator::stepThreadHandler(VoxelGrid *grid ,Forest *, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                stepVoxelHeatTransfer(grid->getVoxel(x, y, z), deltaTime);
            }
        }
    }
}

void Simulator::cleanupForNextStep(VoxelGrid *grid, Forest *forest){
    //No need for asserts here, same asserts as in step()
    int gridResolution = grid->getResolution();
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepCleanupThreadHandler, this, grid, forest, gridResolution, x, x + jumpPerThread);

    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    if (forest != nullptr){ //Forest is optional
        forest->deleteDeadModules();
        forest->updateLastFrameDataOfModules(); //TODO: can we maybe find a way to include this in the multithreading design?
    }
}


void Simulator::stepCleanupThreadHandler(VoxelGrid *grid, Forest *, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                grid->getVoxel(x, y, z)->updateLastFrameData();
            }
        }
    }
}
