#ifndef PHYSICS_H
#define PHYSICS_H
#include "glm/ext.hpp"

//These are all the relavant tunable physics variables for the simulatoin

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
const double viscosity = 0.1;
const double gravity_acceleration = 0.1;
const double air_density = 1.225;

// heat transfer
const double HEAT_DIFFUSION_INTENSITY_TERM = 0.02; //alpha
const double RADIATIVE_COOLING_TERM = 0.01; //gamma
const double adjacent_module_diffusion = 0.04; // alpha_M
const double module_air_diffusion = 0.04; // b

double ambientTemperatureFunc(glm::dvec3 point);

// trees
const double woodDensity = 40;

// fire particle
const double alpha_temp = 0.98;
const double beta_temp = 0.02;
const double burn_coef = 0.1;
const double thermal_expansion = 0.5;

#endif // PHYSICS_H
