#ifndef PHYSICS_H
#define PHYSICS_H

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


#endif // PHYSICS_H
