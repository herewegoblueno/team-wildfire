#include "simulator.h"
#include <thread>
#include <iostream>

//eq 21 in Fire in Paradise paper
//TODO: add in water term
void Simulator::stepVoxelHeatTransfer(Voxel* v, ModuleSet nearbyModules, int deltaTimeInMs){
    VoxelTemperatureGradientInfo tempGradientInfo = v->getTemperatureGradientInfoFromPreviousFrame();

    double differenceFromAmbienceCap = 50;
    //Clamping this because since we raise it to the power 4, larger values can make the tempertature
    //oscilate between very -ve and very +ve until it explodes, similar to
    //https://www.desmos.com/calculator/k86lqwvxvd
    //of course, we woudn't need this if we had a better intergrator or smaller timesteps
    double differenceFromAmbience = clamp(
                v->getLastFrameState()->temperature - v->getAmbientTemperature(),
                -differenceFromAmbienceCap, differenceFromAmbienceCap);
    //Mass loss is considered precomputed
    double dMdt = 0.0;
    for (Module *m : nearbyModules) dMdt += m->getCurrentState()->massChangeRateFromLastFrame;

    double temperature_save = v->getLastFrameState()->temperature;
    // Midpoint Integration
    // first pass
    double dTdt = heat_diffusion_intensity * tempGradientInfo.laplace;
    dTdt -= radiative_cooling * pow(differenceFromAmbience, 4) * ((differenceFromAmbience > 0) ? 1 : -1);
    dTdt -= glm::dot(tempGradientInfo.gradient_pos, v->getLastFrameState()->u)*0.5;
    dTdt -= glm::dot(tempGradientInfo.gradient_neg, v->getNegfaceVel())*0.5;
    dTdt -= module_to_air_diffusion * dMdt;
    v->getLastFrameState()->temperature += 0.5*dTdt * deltaTimeInMs / 1000.0;
    // second pass
    differenceFromAmbience = clamp(
                    v->getLastFrameState()->temperature - v->getAmbientTemperature(),
                    -differenceFromAmbienceCap, differenceFromAmbienceCap);
    tempGradientInfo = v->getTemperatureGradientInfoFromPreviousFrame();
    double dTdt2 = heat_diffusion_intensity * tempGradientInfo.laplace;
    dTdt2 -= radiative_cooling * pow(differenceFromAmbience, 4) * ((differenceFromAmbience > 0) ? 1 : -1);
    dTdt2 -= glm::dot(tempGradientInfo.gradient_pos, v->getLastFrameState()->u)*0.5;
    dTdt2 -= glm::dot(tempGradientInfo.gradient_neg, v->getNegfaceVel())*0.5;
    dTdt2 -= module_to_air_diffusion * dMdt;
    // integrate
    v->getLastFrameState()->temperature = temperature_save;
    v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt2 * deltaTimeInMs / 1000.0;
    v->getCurrentState()->tempGradientFromPrevState = tempGradientInfo.gradient;
    v->getCurrentState()->tempLaplaceFromPrevState = tempGradientInfo.laplace;

//    if(std::abs(v->getCurrentState()->temperature)>10000)
//    {
//        dvec3 u = v->getLastFrameState()->u;
////        cout<< "[" << v->XIndex << "," << v->YIndex << "," << v->ZIndex << ",(";
////        cout<< u.x <<"," <<u.y<<"," <<u.z << "]  " << std::flush;
//    }
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
            + air_to_module_diffusion * (surroundingAirTemp - moduleTemp);
    m->getCurrentState()->temperature = moduleTemp + dTdt * deltaTimeInMs / 1000.0;
};
