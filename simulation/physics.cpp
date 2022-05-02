#include "glm/ext.hpp"
#include "physics.h"
#include "voxels/voxelgrid.h"

double ambientTemperatureFunc(glm::dvec3 point){
    return glm::clamp(3.5 - point.y / 2, 0.0, 3.5 );
}

double maxReasonableCelcuis = 200;
double minReasonableCelcuis = 20;
double celciusdiff = maxReasonableCelcuis - minReasonableCelcuis;

double maxSimulationTemp = 20;
double minSimulationTemp = 2;
double simDiff = maxSimulationTemp - minSimulationTemp;

double simTempToWorldTemp(double simTemp){
    return minReasonableCelcuis + ((simTemp - minSimulationTemp) / simDiff) * celciusdiff;
}

double worldTempToSimulationTemp(double worldTemp){
    return minSimulationTemp + ((worldTemp - minReasonableCelcuis) / celciusdiff) * simDiff;
}

