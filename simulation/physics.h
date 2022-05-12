#ifndef PHYSICS_H
#define PHYSICS_H
#include "glm/ext.hpp"

// These are all the relevant tunable physics variables for the simulation

// general
const double sealevel_temperature = 20;
const double sealevel_pressure = 100000;
const double mass_scale = 1;
const double height_scale = 1000;

// water component related
const double autoconverge_cloud = 0.001; // alpha A
const double raindrop_accelerate = 0.01; // K c
const double evaporation_rate = 0.0001; // w

// wind component related
const double vorticity_epsilon = 0.1;
const double viscosity = 0.01;
const double gravity_acceleration = 1;
const double air_density = 1.225;

// heat transfer
const double heat_diffusion_intensity = 0.3; //alpha

#ifdef CUDA_FLUID
    const double radiative_cooling = 0.0002; //gamma
#else
    const double radiative_cooling = 0.0004; //gamma
#endif
const double adjacent_module_diffusion = 0.16; // alpha_M
const double air_to_module_diffusion = 0.75; // b
const double module_to_air_diffusion = 200; // tau

// combustion
const double min_combust_temp_cel = 150.0; // T_0
const double max_combust_temp_cel = 450.0; // T_1
const double max_wind_combustion_boost = 1.5; // n_max
const double speed_for_max_wind_boost = 1.0; // u_ref

const double vapor_release_ratio = 100;
const double reation_rate_multiplier = 1.0; //Not in paper, added by us


// trees
const double woodDensity = 100; // rho

// voxels
double ambientTemperatureFunc(glm::dvec3 point);
// when calculating gradients, we act as if the cells are larger than they actually are
// for better simulation stability
const double voxelSizeMultiplierForGradients = 10.0;

// fire particle
const double alpha_temp = 0.96;
const double beta_temp = 0.04;
const double burn_coef = 0.1;
const double thermal_expansion = 0.05;

//For temperature conversions
const double maxReasonableCelcuis = 600;
const double minReasonableCelcuis = 20;
const double celciusdiff = maxReasonableCelcuis - minReasonableCelcuis;

const double maxSimulationTemp = 20;
const double minSimulationTemp = 2;
const double simDiff = maxSimulationTemp - minSimulationTemp;

double worldTempToSimulationTemp(double worldTemp);
double simTempToWorldTemp(double simTemp);


#endif // PHYSICS_H
