#include "simulator.h"
#include <thread>
#include <iostream>
#include "support/Settings.h"

//eq 21 in Fire in Paradise paper
//(the water term is handled in the water physics part of the simulation)
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

    // Vapor release (the rest of the workon q_v will be done in the dedicated water physics part of the simulation)
    v->getLastFrameState()->q_v -= dMdt*vapor_release_ratio*deltaTimeInMs/1000.0;
    v->getLastFrameState()->q_v = clamp(v->getLastFrameState()->q_v, 0.f, 1.f);

    // Midpoint Integration
    // first pass
    double temperature_save = v->getLastFrameState()->temperature;
    double dTdt = heat_diffusion_intensity * tempGradientInfo.laplace;
    dTdt -= radiative_cooling * pow(differenceFromAmbience, 4) * ((differenceFromAmbience > 0) ? 1 : -1);
    dTdt -= glm::dot(tempGradientInfo.gradient_pos, v->getLastFrameState()->u)*0.5;
    dTdt -= glm::dot(tempGradientInfo.gradient_neg, v->getNegfaceVel())*0.5;
    dTdt -= module_to_air_diffusion * dMdt;

    if (settings.useMidpointForVoxelHeatTransfer){
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
    }else{
        v->getCurrentState()->temperature = v->getLastFrameState()->temperature + dTdt * deltaTimeInMs / 1000.0;
    }

    if(std::abs(v->getCurrentState()->temperature) > 100) {
        cout << "Looks like a voxel's temperature is exploding! " << v->getCurrentState()->temperature << endl;
    }

    v->getCurrentState()->tempGradientFromPrevState = tempGradientInfo.gradient;
    v->getCurrentState()->tempLaplaceFromPrevState = tempGradientInfo.laplace;

};

/** Equation 25 of Fire in Paradise paper */
void Simulator::stepModuleHeatTransfer(Module *m, VoxelSet surroundingAir, int deltaTimeInMs) {
    double moduleTemp = m->getLastFrameState()->temperature;
    double surroundingAirTemp;
    if (surroundingAir.size() > 0) {
        surroundingAirTemp = 0.0;
        for (Voxel *v : surroundingAir) {
            surroundingAirTemp += v->getLastFrameState()->temperature;
        }
        surroundingAirTemp = surroundingAirTemp / static_cast<double>(surroundingAir.size());
    } else {
        // If we don't have any surrounding voxels, it's because the burning module is
        // very thin and about to get deleted. Using the ambient temp here will speed up
        // the burning, but it's ok since the module will likely be gone in a few frames.
        surroundingAirTemp = ambientTemperatureFunc(m->getCenterOfMass());
    }

    double tempLaplace = m->getTemperatureLaplaceFromPreviousFrame();
    double dTdt = adjacent_module_diffusion * tempLaplace
            + air_to_module_diffusion * (surroundingAirTemp - moduleTemp);
    m->getCurrentState()->temperature = moduleTemp + dTdt * deltaTimeInMs / 1000.0;
};
