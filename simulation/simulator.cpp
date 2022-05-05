#include "simulator.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <memory.h>
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
        forest->updateMassAndAreaOfModulesViaBurning(deltaTime);
        //TODO: refactor a bit (better encapsulation maybe)
        for (Module *m : forest->getModules()) {
            VoxelSet surroundingAir = forest->getVoxelsMappedToModule(m);
            stepModuleHeatTransfer(m, surroundingAir, deltaTime);
        }
        //<TODO: water content of modules should go here>
        forest->updateMassOfVoxels();
    }

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

//TODO: eventually clean up, this is just here for testing
void Simulator::linear_step(VoxelGrid *grid, Forest *forest)
{
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    double deltaTime = (currentTime - timeLastFrame).count();
    if (deltaTime > 100) deltaTime = 100;

    deltaTime = 10;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    int face_num = gridResolution*gridResolution;
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    int cell_num = gridResolution*gridResolution*gridResolution;
    double* grid_temp = (double *) malloc(cell_num*sizeof(double));
    double* grid_q_v = (double *) malloc(cell_num*sizeof(double));
    double* grid_h = (double *) malloc(cell_num*sizeof(double));
    double* u_xyz = (double *) malloc(cell_num*sizeof(double)*3);
    int* id_xyz = (int *) malloc(cell_num*sizeof(int)*3);

    //(lines 7-12 in Algorithm 1 of paper)
    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWind, this, grid, forest, deltaTime, gridResolution,
                             x, x + jumpPerThread, grid_temp, grid_q_v, grid_h , u_xyz, id_xyz);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    processWindGPU(grid_temp, grid_q_v, grid_h, u_xyz, id_xyz, 20,
                   gridResolution, grid->cellSideLength(), deltaTime/1000.);

    threads.clear();
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWater, this, grid, forest, deltaTime,
                             gridResolution, x, x + jumpPerThread, u_xyz);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    free(grid_temp);free(grid_q_v);free(grid_h);free(u_xyz);free(id_xyz);

}

void Simulator::stepThreadHandler(VoxelGrid *grid ,Forest * forest, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                Voxel *v = grid->getVoxel(x, y, z);
                ModuleSet nearbyModules = forest == nullptr ? ModuleSet() : forest->getModulesMappedToVoxel(v);
                stepVoxelHeatTransfer(v, nearbyModules, deltaTime);
            }
        }
    }
}

void Simulator::stepThreadHandlerWind(VoxelGrid *grid, Forest *forest, double deltaTime, int resolution,
                                      int minXInclusive, int maxXExclusive,
                                      double* grid_temp, double* grid_q_v, double* grid_h , double* u_xyz, int* id_xyz)
{
//    double cell_size = grid->cellSideLength();
//    double density_term = calc_density_term(cell_size, deltaTime);
    int index, face_num = resolution*resolution;
//    int cell_num = face_num*resolution;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*face_num;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel *v = grid->getVoxel(x, y, z);
                ModuleSet nearbyModules = forest == nullptr ? ModuleSet() : forest->getModulesMappedToVoxel(v);
                stepVoxelHeatTransfer(v, nearbyModules, deltaTime);

            #ifdef CUDA_FLUID
                grid_temp[index] = v->getCurrentState()->temperature;
                grid_q_v[index] = v->getLastFrameState()->q_v;
                grid_h[index] = v->centerInWorldSpace.y;
                u_xyz[index*3+0] = v->getLastFrameState()->u.x;
                u_xyz[index*3+1] = v->getLastFrameState()->u.y;
                u_xyz[index*3+2] = v->getLastFrameState()->u.z;
                id_xyz[index*3+0] = v->XIndex;
                id_xyz[index*3+1] = v->YIndex;
                id_xyz[index*3+2] = v->ZIndex;
            #endif
                index++;
            }
        }
    }
}

void Simulator::stepThreadHandlerWater(VoxelGrid *grid ,Forest *, double deltaTime, int resolution,
                                       int minXInclusive, int maxXExclusive, double* u_new){
    double cell_size = grid->cellSideLength();
    int index;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*resolution*resolution;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel* vox = grid->getVoxel(x,y,z);

            #ifdef CUDA_FLUID
                dvec3 u(u_new[index*3], u_new[index*3+1], u_new[index*3+2]);
                vox->getCurrentState()->u = u;
            #endif
//                stepVoxelWater(vox, deltaTime);
                index++;
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
