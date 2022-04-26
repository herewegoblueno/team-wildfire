#include "glm/ext.hpp"
#include "physics.h"
#include "voxels/voxelgrid.h"

double ambientTemperatureFunc(glm::dvec3 point){
    return glm::clamp(3.5 - point.y / 2, 0.0, 3.5 );
}
