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

    deltaTime = 0.0001;
    timeLastFrame = currentTime;
    if (deltaTime == 0) return; //Don't bother doing anything


    int gridResolution = grid->getResolution();
    int face_num = gridResolution*gridResolution;
    assert(gridResolution % NUMBER_OF_SIMULATION_THREADS == 0);
    int jumpPerThread = gridResolution / NUMBER_OF_SIMULATION_THREADS;

    int cell_num = gridResolution*gridResolution*gridResolution;
    double* diag = (double *) malloc(cell_num*sizeof(double));
    double* rhs = (double *) malloc(cell_num*sizeof(double));
    int* id_xyz = (int *) malloc(cell_num*sizeof(int)*3);

    //(lines 7-12 in Algorithm 1 of paper)
    std::vector<std::thread> threads;
    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWind, this, grid, forest, deltaTime,
                             gridResolution, x, x + jumpPerThread, diag, rhs, id_xyz);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate
    double pressure[10][10][10];
    for (int x=0;x<10;x++)
        for (int y=0;y<10;y++)
            for (int z=0;z<10;z++)
                pressure[x][y][z] = rhs[x*face_num + y*gridResolution + z];

    threads.clear();
    pressure_projection_Jacobi_cuda(diag, rhs, id_xyz, gridResolution, cell_num, 20);
    for (int x=0;x<10;x++)
        for (int y=0;y<10;y++)
            for (int z=0;z<10;z++)
                pressure[x][y][z] = rhs[x*face_num + y*gridResolution + z];

    for (int x = 0; x < gridResolution; x += jumpPerThread)
        threads.emplace_back(&Simulator::stepThreadHandlerWater, this, grid, forest, deltaTime,
                             gridResolution, x, x + jumpPerThread, rhs);
    for (auto& th : threads) th.join();  //Wait for all the threads to terminate

    free(diag);free(rhs);free(id_xyz);

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
                                      int minXInclusive, int maxXExclusive, double* diag_A, double* rhs, int* id_xyz)
{
    double cell_size = grid->cellSideLength();
    double density_term = calc_density_term(cell_size, deltaTime);
    int index, face_num = resolution*resolution;
//    int cell_num = face_num*resolution;
    for (int x = minXInclusive; x < maxXExclusive; x++){
        index = x*face_num;
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel *v = grid->getVoxel(x, y, z);
                ModuleSet nearbyModules = forest == nullptr ? ModuleSet() : forest->getModulesMappedToVoxel(v);
                stepVoxelHeatTransfer(v, nearbyModules, deltaTime);
                stepVoxelWind(grid->getVoxel(x, y, z), deltaTime);

            #ifdef CUDA_FLUID
                fill_jacobi_rhs(grid->getVoxel(x,y,z), resolution, index, density_term,
                                     diag_A+index, rhs+index, id_xyz+index*3);
            #endif
                index++;
            }
        }
    }
}

void Simulator::stepThreadHandlerWater(VoxelGrid *grid ,Forest *, double deltaTime, int resolution,
                                       int minXInclusive, int maxXExclusive, double* pressure){
    double cell_size = grid->cellSideLength();
    for (int x = minXInclusive; x < maxXExclusive; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel* vox = grid->getVoxel(x,y,z);

            #ifdef CUDA_FLUID
                dvec3 d_u = calc_pressure_effect(x, y, z, resolution, pressure, deltaTime, cell_size);
                dvec3 o_u = vox->getCurrentState()->u;
                if(x==20 && y==20 && z==20)
                {
                    cout << "[" << d_u.x << "," << d_u.y << "," << d_u.z << "]-";
                    cout << "[" << o_u.x << "," << o_u.y << "," << o_u.z << "]\n";
                }
                vox->getCurrentState()->u = o_u - d_u;
            #endif
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
