#include "simulator.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <memory.h>
#include "support/Settings.h"


const int Simulator::NUMBER_OF_SIMULATION_THREADS = 8;

Simulator::Simulator() {}

void Simulator::init(){
    timeLastFrame = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
}

void Simulator::step(VoxelGrid *grid, Forest *forest){
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    int deltaTime = (currentTime - timeLastFrame).count();
    if (deltaTime > 100) deltaTime = 100;
    deltaTime *= settings.simulatorTimescale;
    timeSinceLastFrame = deltaTime;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    mallocHost2cuda(grid);
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

#ifdef CUDA_FLUID
    dvec3 g_w = grid->getGlobalFField();
    double g_w3[3] = {g_w.x, g_w.y, g_w.z};
    processWindGPU(host2cuda.grid_temp, host2cuda.grid_q_v, host2cuda.grid_h, host2cuda.u_xyz, host2cuda.id_xyz,
                   64, g_w3, gridResolution, grid->cellSideLengthForGradients(), deltaTime/1000.);
    threads.clear();
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepCuda2hostThreadHandler, this, grid, forest, deltaTime, gridResolution, x, x + jumpPerThread);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate
#endif

    if (forest != nullptr){ //Forest is optional
        //This should be the last step of the simulation
        //We want the actual step to be based on last frame's mapping, but we want the scene's rendering
        //(which starts after the step function) to be based on the resultant mapping after the step
        //(so the users see the most up to date state)
        forest->updateModuleVoxelMapping();
    }
    freeHost2cuda();
}



void Simulator::stepThreadHandler(VoxelGrid *grid ,Forest * forest, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                Voxel *v = grid->getVoxel(x, y, z);
                ModuleSet nearbyModules = forest == nullptr ? ModuleSet() : forest->getModulesMappedToVoxel(v);
                stepVoxelHeatTransfer(v, nearbyModules, deltaTime);
                writeHost2cudaSpace(v, x*resolution*resolution+y*resolution+z);
            }
        }
    }
}

void Simulator::stepCuda2hostThreadHandler(VoxelGrid *grid ,Forest * forest, int deltaTime, int resolution,
                                           int minXInclusive, int maxXExclusive){
    int index;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*resolution*resolution;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel* vox = grid->getVoxel(x,y,z);
                dvec3 u(host2cuda.u_xyz[index*3], host2cuda.u_xyz[index*3+1], host2cuda.u_xyz[index*3+2]);
                vox->getCurrentState()->u = u;
                stepVoxelWater(vox, deltaTime/1000.);
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

float Simulator::getTimeSinceLastFrame()
{
    return timeSinceLastFrame;
}

void Simulator::mallocHost2cuda(VoxelGrid *grid)
{
#ifdef CUDA_FLUID
    int gridResolution = grid->getResolution();
    int cell_num = gridResolution*gridResolution*gridResolution;
    host2cuda.grid_temp = (double *) malloc(cell_num*sizeof(double));
    host2cuda.grid_q_v = (double *) malloc(cell_num*sizeof(double));
    host2cuda.grid_h = (double *) malloc(cell_num*sizeof(double));
    host2cuda.u_xyz = (double *) malloc(cell_num*sizeof(double)*3);
    host2cuda.id_xyz = (int *) malloc(cell_num*sizeof(int)*3);
#endif
}

void Simulator::writeHost2cudaSpace(Voxel* v, int index)
{
#ifdef CUDA_FLUID
    host2cuda.grid_temp[index] = v->getCurrentState()->temperature;
    host2cuda.grid_q_v[index] = v->getLastFrameState()->q_v;
    host2cuda.grid_h[index] = v->centerInWorldSpace.y;
    host2cuda.u_xyz[index*3+0] = v->getLastFrameState()->u.x;
    host2cuda.u_xyz[index*3+1] = v->getLastFrameState()->u.y;
    host2cuda.u_xyz[index*3+2] = v->getLastFrameState()->u.z;
    host2cuda.id_xyz[index*3+0] = v->XIndex;
    host2cuda.id_xyz[index*3+1] = v->YIndex;
    host2cuda.id_xyz[index*3+2] = v->ZIndex;
#endif
}

void Simulator::freeHost2cuda()
{
#ifdef CUDA_FLUID
    free(host2cuda.grid_temp);
    free(host2cuda.grid_q_v);
    free(host2cuda.grid_h);
    free(host2cuda.u_xyz);
    free(host2cuda.id_xyz);
#endif
}
