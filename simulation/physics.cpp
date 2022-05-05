#include "glm/ext.hpp"
#include "physics.h"
#include "voxels/voxelgrid.h"

double ambientTemperatureFunc(glm::dvec3 point){
    return 3.5 - point.y*0.3;
}

double simTempToWorldTemp(double simTemp){
    return minReasonableCelcuis + ((simTemp - minSimulationTemp) / simDiff) * celciusdiff;
}

double worldTempToSimulationTemp(double worldTemp){
    return minSimulationTemp + ((worldTemp - minReasonableCelcuis) / celciusdiff) * simDiff;
}

