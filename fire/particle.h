#ifndef PARTICLE_H
#define PARTICLE_H

#include <memory>
#include <vector>
#include <random>

#include "GL/glew.h"
#include <glm.hpp>


struct Particle {
    glm::vec3 Position, Velocity;
    glm::vec4 Color;
    float     Life;
    float     Temp;

    Particle()
      : Position(0.0f), Velocity(0.0f), Color(1.0f), Life(0.0f) { }
};


const float alpha_temp = 0.98;
const float beta_temp = 0.02;
const float burn_coef = 0.1;

const float thermal_expansion = 0.005;
const glm::vec3 gravity(0,-1,0);


#endif
