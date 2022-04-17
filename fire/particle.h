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

    Particle()
      : Position(0.0f), Velocity(0.0f), Color(1.0f), Life(0.0f) { }
};



#endif
