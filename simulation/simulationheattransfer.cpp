#include "simulator.h"
#include <thread>

 const float Simulator::HEAT_DIFFUSION_INTENSITY_TERM = 0.1; //alpha
 const float Simulator::RADIATIVE_COOLING_TERM = 0.03; //gamma

//eq 21 in Fire in Paradise paper
//TODO: add in last 2 terms
void Simulator::stepVoxelHeatTransfer(Voxel* v, int deltaTimeInMs){
    vec3 tempGradient = v->getTemperatureGradientFromPreviousFrame();
    v->getCurrentState()->tempGradientFromPrevState = tempGradient;
    float dTdt = HEAT_DIFFUSION_INTENSITY_TERM * glm::dot(tempGradient, tempGradient);
    dTdt -= RADIATIVE_COOLING_TERM * pow(v->getLastFrameState()->temperature - Voxel::getAmbientTemperature(v->centerInWorldSpace), 4);
    dTdt -= glm::dot(tempGradient, v->getLastFrameState()->u);
    v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt * deltaTimeInMs / 1000.f;
};


