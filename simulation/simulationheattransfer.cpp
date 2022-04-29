#include "simulator.h"
#include <thread>

//eq 21 in Fire in Paradise paper
//TODO: add in last 2 terms
void Simulator::stepVoxelHeatTransfer(Voxel* v, int deltaTimeInMs){

    VoxelTemperatureGradientInfo tempGradientInfo = v->getTemperatureGradientInfoFromPreviousFrame();
    v->getCurrentState()->tempGradientFromPrevState = tempGradientInfo.gradient;
    v->getCurrentState()->tempLaplaceFromPrevState = tempGradientInfo.laplace;

    double dTdt = HEAT_DIFFUSION_INTENSITY_TERM * tempGradientInfo.laplace;
    double differenceFromAmbience = v->getLastFrameState()->temperature - v->getAmbientTemperature();
    dTdt -= RADIATIVE_COOLING_TERM * pow(differenceFromAmbience, 4) * ((differenceFromAmbience > 0) ? 1 : -1);
    dTdt -= glm::dot(tempGradientInfo.gradient, v->getLastFrameState()->u);

    v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt * deltaTimeInMs / 1000.0;
};

/** Equation 25 of Fire in Paradise paper */
void Simulator::stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs) {
    double moduleTemp = m->getLastFrameState()->temperature;
    double tempLaplace = m->getTemperatureLaplaceFromPreviousFrame();
    double surroundingAirTemp = 0.0;
    for (Voxel *v : surroundingAir) {
        surroundingAirTemp += v->getLastFrameState()->temperature;
    }
    surroundingAirTemp = surroundingAirTemp / static_cast<double>(surroundingAir.size());
    double dTdt = adjacent_module_diffusion * tempLaplace
            + module_air_diffusion * (surroundingAirTemp - moduleTemp);
    m->getCurrentState()->temperature = moduleTemp + dTdt * deltaTimeInMs / 1000.0;
};
