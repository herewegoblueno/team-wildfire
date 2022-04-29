#ifndef PHYSICS_H
#define PHYSICS_H
#include "glm/ext.hpp"

//These are all the relevant tunable physics variables for the simulation

// general
const float sealevel_temperature = 280;
const float sealevel_pressure = 1;
const float mass_scale = 1;
const float height_scale = 1000;

// water component related
const float autoconverge_cloud = 0.001; // alpha A
const float raindrop_accelerate = 0.01; // K c
const float evaporation_rate = 0.001; // w

// wind component related
const float verticity_epsilon = 0.001;
const float gravity_acceleration = 9.8;

// heat transfer
const double HEAT_DIFFUSION_INTENSITY_TERM = 0.02; //alpha
const double RADIATIVE_COOLING_TERM = 0.01; //gamma
const double adjacent_module_diffusion = 0.04; // alpha_M
const double module_air_diffusion = 0.04; // b

// combustion
const double reaction_rate_t0 = 2.0; // TODO: make these physically accurate
const double reaction_rate_t1 = 3.5;
const double max_wind_combustion_boost = 1.5; // n_max
const double speed_for_max_wind_boost = 1.0; // u_ref
const double reation_rate_multiplier = 1.5; //Not in paper, added by us

//Voxels
double ambientTemperatureFunc(glm::dvec3 point);

// trees
const float woodDensity = 40;

#endif // PHYSICS_H
