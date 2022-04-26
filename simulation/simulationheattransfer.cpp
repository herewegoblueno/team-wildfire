#include "simulator.h"
#include <thread>

//eq 21 in Fire in Paradise paper
//TODO: add in last 2 terms
void Simulator::stepVoxelHeatTransfer(Voxel* v, int deltaTimeInMs){
    VoxelTemperatureGradientInfo tempGradientInfo = v->getTemperatureGradientInfoFromPreviousFrame();
    v->getCurrentState()->tempGradientFromPrevState = tempGradientInfo.gradient;
    v->getCurrentState()->tempLaplaceFromPrevState = tempGradientInfo.laplace;

    double dTdt = HEAT_DIFFUSION_INTENSITY_TERM * tempGradientInfo.laplace;
    dTdt -= RADIATIVE_COOLING_TERM * pow(v->getLastFrameState()->temperature - v->getAmbientTemperature(), 4);
    dTdt -= glm::dot(tempGradientInfo.gradient, v->getLastFrameState()->u);

    v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt * deltaTimeInMs / 1000.0;
};


