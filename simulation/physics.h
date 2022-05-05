#ifndef PHYSICS_H
#define PHYSICS_H
#include "glm/ext.hpp"

//These are all the relevant tunable physics variables for the simulation

// general
const double sealevel_temperature = 280;
const double sealevel_pressure = 100000;
const double mass_scale = 1;
const double height_scale = 1000;

// water component related
const double autoconverge_cloud = 0.001; // alpha A
const double raindrop_accelerate = 0.01; // K c
const double evaporation_rate = 0.001; // w

// wind component related
const double vorticity_epsilon = 0.1;
const double viscosity = 0.01;
const double gravity_acceleration = 1;
const double air_density = 1.225;

// heat transfer
const double HEAT_DIFFUSION_INTENSITY_TERM = 0.2; //alpha
const double RADIATIVE_COOLING_TERM = 0.002; //gamma
const double adjacent_module_diffusion = 0.08; // alpha_M
const double air_to_module_diffusion = 0.75; // b
const double module_to_air_diffusion = 150; // tau

// combustion
const double min_combust_temp_cel = 150.0; // T_0
const double max_combust_temp_cel = 450.0; // T_1
const double max_wind_combustion_boost = 1.5; // n_max
const double speed_for_max_wind_boost = 1.0; // u_ref
const double reation_rate_multiplier = 1.0; //Not in paper, added by us

//Voxels
double ambientTemperatureFunc(glm::dvec3 point);

// trees
const double woodDensity = 40;

// fire particle
const double alpha_temp = 0.9;
const double beta_temp = 0.1;
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
