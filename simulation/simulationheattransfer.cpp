#include "simulator.h"
#include <thread>
#include <iostream>

//eq 21 in Fire in Paradise paper
//TODO: add in last 2 terms
void Simulator::stepVoxelHeatTransfer(Voxel* v, ModuleSet nearbyModules, int deltaTimeInMs){
    VoxelTemperatureGradientInfo tempGradientInfo = v->getTemperatureGradientInfoFromPreviousFrame();
    v->getCurrentState()->tempGradientFromPrevState = tempGradientInfo.gradient;
    v->getCurrentState()->tempLaplaceFromPrevState = tempGradientInfo.laplace;

    double dTdt = HEAT_DIFFUSION_INTENSITY_TERM * tempGradientInfo.laplace;
    double differenceFromAmbienceCap = 50;
    //Clamping this because since we raise it to the power 4, larger values can make the tempertature
    //oscilate between very -ve and very +ve until it explodes, similar to
    //https://www.desmos.com/calculator/k86lqwvxvd
    //of course, we woudn't need this if we had a better intergrator or smaller timesteps
    double differenceFromAmbience = clamp(
                v->getLastFrameState()->temperature - v->getAmbientTemperature(),
                -differenceFromAmbienceCap, differenceFromAmbienceCap);
    dTdt -= RADIATIVE_COOLING_TERM * pow(differenceFromAmbience, 4) * ((differenceFromAmbience > 0) ? 1 : -1);

    //TODO: this should be based on the iterpolated u field (to get the u field in the center),
    //not the u field the voxel is storing (which is relative to its faces)
    dTdt -= glm::dot(tempGradientInfo.gradient, v->getLastFrameState()->u);
    if(std::abs(dTdt)>20000)
    {
        cout<< "dT error";
    }

    
    double dMdt = 0.0;
    for (Module *m : nearbyModules) {
        dMdt += m->getCurrentState()->massChangeRateFromLastFrame;
    }
    dTdt -= module_to_air_diffusion * dMdt;
    v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt * deltaTimeInMs / 1000.0;
    if(std::abs(v->getCurrentState()->temperature)>20000)
    {
        cout<< "accum error";
    }
    if (v->XIndex==18 && v->YIndex==18 && v->ZIndex==18) v->getCurrentState()->temperature=15;
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
    double dTdt = 100 * tempLaplace
            + air_to_module_diffusion * (surroundingAirTemp - moduleTemp);
    m->getCurrentState()->temperature = moduleTemp + dTdt * deltaTimeInMs / 1000.0;
};
