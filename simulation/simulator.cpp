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

    deltaTime = 0.001;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    int cell_num = gridResolution*gridResolution*gridResolution;
    double* diag = (double *) malloc(cell_num*sizeof(double));
    int* id_xyz = (int *) malloc(cell_num*sizeof(int)*3);
    double* rhs = (double *) malloc(cell_num*sizeof(double));

    //(lines 7-12 in Algorithm 1 of paper)
    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWind, this, grid, forest, deltaTime,
                             gridResolution, x, x + jumpPerThread, diag, rhs, id_xyz);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    threads.clear();
    pressure_projection_Jacobi_cuda(diag, rhs, id_xyz, gridResolution, cell_num, gridResolution*gridResolution);
    free(diag);
    free(id_xyz);
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWater, this, grid, forest, deltaTime, gridResolution, x, x + jumpPerThread, rhs);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate
    free(rhs);
}

void Simulator::stepThreadHandler(VoxelGrid *grid ,Forest *, int deltaTime, int resolution, int minXInclusive, int maxXExclusive){
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
                                      int minXInclusive, int maxXExclusive, double* diag_A, double* rhs, int* id_xyz)
{
    double cell_size = grid->cellSideLength();
    double density_term = deltaTime/air_density/cell_size/cell_size;
    int index, face_num = resolution*resolution;
//    int cell_num = face_num*resolution;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*face_num;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                stepVoxelHeatTransfer(grid->getVoxel(x, y, z), deltaTime*1000);
                stepVoxelWind(grid->getVoxel(x, y, z), deltaTime);

                double diag = 6;
                glm::dvec3 gradient = grid->getVoxel(x,y,z)->getVelGradient();
                double this_rhs = (gradient.x + gradient.y + gradient.z)/density_term;

                if(x>resolution-2) diag --;
                if(x<1) diag --;
//                if(y>resolution-2) diag --;
                if(y<1) diag --;
                if(z>resolution-2) diag --;
                if(z<1) diag --;

                diag_A[index] = diag;
                int* i_xyz = id_xyz+index*3;
                i_xyz[0] = x;
                i_xyz[1] = y;
                i_xyz[2] = z;

                assert(!std::isnan(this_rhs));
                rhs[index] = this_rhs;
                index++;
            }
        }
    }
}

void Simulator::stepThreadHandlerWater(VoxelGrid *grid ,Forest *, double deltaTime, int resolution,
                                       int minXInclusive, int maxXExclusive, double* pressure){
    int face_num = resolution*resolution;
    double cell_size = grid->cellSideLength();
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                glm::dvec3 deltaP(0,0,0);
                int index = x*face_num+y*resolution+z;
                if(x<resolution-1) deltaP.x += pressure[index+face_num];
                else deltaP.x += pressure[index];
                if(x>0) deltaP.x -= pressure[index-face_num];
                else deltaP.x -= pressure[index];

                if(y<resolution-1) deltaP.y += pressure[index+resolution];
                else deltaP.y += pressure[index];
                if(y>0) deltaP.y -= pressure[index-resolution];
                else deltaP.y -= pressure[index];

                if(z<resolution-1) deltaP.z += pressure[index+1];
                else deltaP.z += pressure[index];
                if(z>0) deltaP.z -= pressure[index-1];
                else deltaP.z -= pressure[index];
                Voxel* vox = grid->getVoxel(x,y,z);
                vox->getCurrentState()->u -= deltaP*(double)deltaTime/cell_size/air_density;
                if(glm::length(vox->getCurrentState()->u)>100)
                {
                    std::cout << "[large vel after pressure]";
                }
//                stepVoxelWater(vox, deltaTime);
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
