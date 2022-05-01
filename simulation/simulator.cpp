#include "simulator.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <memory.h>
#include "support/Settings.h"
extern "C"
void processWindGPU(double* grid_temp, double* grid_q_v, double* grid_h, int resolution);

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


void Simulator::linear_step(VoxelGrid *grid, Forest *forest)
{
    milliseconds currentTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    int deltaTime = (currentTime - timeLastFrame).count();
    if (deltaTime > 100) deltaTime = 100;
    deltaTime *= settings.simulatorTimescale;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    int cell_num = gridResolution*gridResolution*gridResolution;
    double* mat_A = (double *) malloc(cell_num*cell_num*sizeof(double));
    double* dvg = (double *) malloc(cell_num*sizeof(double));

    memset(mat_A, 0, cell_num*cell_num*sizeof(double));
    memset(dvg, 0, cell_num*sizeof(double));

    //(lines 7-12 in Algorithm 1 of paper)
    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWind, this, grid, forest, deltaTime,
                             gridResolution, x, x + jumpPerThread, mat_A, dvg);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    threads.clear();
    pressure_projection_Jacobi_cuda(mat_A, dvg, cell_num*cell_num, cell_num, 20);
    free(mat_A);
    free(dvg);
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWater, this, grid, forest, deltaTime, gridResolution, x, x + jumpPerThread);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate
}

void Simulator::stepThreadHandler(VoxelGrid *grid ,Forest *, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                stepVoxelHeatTransfer(grid->getVoxel(x, y, z), deltaTime);
                stepVoxelWind(grid->getVoxel(x, y, z), deltaTime);
            }
        }
    }
}

void Simulator::stepThreadHandlerWind(VoxelGrid *grid, Forest *forest, int deltaTime, int resolution,
                                      int minXInclusive, int maxXExclusive, double* mat_A, double* dvg)
{
    double cell_size = grid->cellSideLength();
    double density_term = deltaTime/1/cell_size/cell_size;
    int index, face_num = resolution*resolution;
    int cell_num = face_num*resolution;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*face_num;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                stepVoxelHeatTransfer(grid->getVoxel(x, y, z), deltaTime);
                stepVoxelWind(grid->getVoxel(x, y, z), deltaTime);


                double diag = 6;

                if(x<resolution-1) mat_A[index*cell_num + index+face_num] = -1;
                else diag --;
                if(x>0) mat_A[index*cell_num +  index-face_num] = -1;
                else diag --;
                if(y<resolution-1) mat_A[index*cell_num +  index+resolution] = -1;
                else diag --;
                if(y>0) mat_A[index*cell_num + index-resolution] = -1;
                else diag --;
                if(z<resolution-1) mat_A[index*cell_num + index+1] = -1;
                else diag --;
                if(z>0) mat_A[index*cell_num + index-1] = -1;
                else diag --;

                mat_A[index*cell_num + index] = diag;

                glm::dvec3 gradient = grid->getVoxel(x,y,z)->getVelGradient();
                dvg[index] = (gradient.x + gradient.y + gradient.z)/density_term;
            }
        }
    }
}

void Simulator::stepThreadHandlerWater(VoxelGrid *grid ,Forest *, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                //<TODO: voxel water updates should go here>
                stepVoxelWater(grid->getVoxel(x, y, z), deltaTime);
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
