#include "voxel.h"
#include "glm/ext.hpp"
#define GLM_ENABLE_EXPERIMENTAL


Voxel::Voxel(VoxelGrid *grid, int XIndex, int YIndex, int ZIndex, vec3 center) :
    grid(grid),
    XIndex(XIndex),
    YIndex(YIndex),
    ZIndex(ZIndex),
    centerInWorldSpace(center)
{
    lastFramePhysicalState.temperature = getAmbientTemperature(center);
    currentPhysicalState.temperature = getAmbientTemperature(center);

    //for testing
    int targetIndex = ceil(grid->getResolution() / 2.f);
    int size = 4;
    if (XIndex > targetIndex - size && XIndex < targetIndex + size &&
            YIndex > targetIndex - size && YIndex < targetIndex + size &&
            ZIndex > targetIndex - size && ZIndex < targetIndex + size){
        lastFramePhysicalState.temperature = 10;
        currentPhysicalState.temperature = 10;
    }
}

//TODO: consider memoization for immediate neighbours
//maybe use an array like Voxel *immediateNeighbours[9] = {[0 ... 8] = nullptr};
//and initialize it for every voxel after each voxel has been made once
Voxel *Voxel::getVoxelWithIndexOffset(vec3 offset){
    return grid->getVoxel(XIndex + offset.x, YIndex + offset.y, ZIndex + offset.z);
}

VoxelPhysicalData *Voxel::getLastFrameState(){
    return &lastFramePhysicalState;
}

VoxelPhysicalData *Voxel::getCurrentState(){
    return &currentPhysicalState;
}

void Voxel::updateLastFrameData(){
    lastFramePhysicalState = currentPhysicalState;
}


double Voxel::getAmbientTemperature(vec3 pos){
    return glm::clamp(2.0 - pos.y / 4.0, 0.0, 2.0 );
}


double Voxel::getNeighbourTemperature(int xOffset, int yOffset, int zOffset){
    Voxel *vox = getVoxelWithIndexOffset({xOffset, yOffset, zOffset});
    //TODO: improve this! boundaries should not be 0
     double temp = vox == nullptr ? 0 : vox->getLastFrameState()->temperature;
     return temp;
}

 VoxelTemperatureGradientInfo Voxel::getTemperatureGradientInfoFromPreviousFrame(){
    double temperatureTop = getNeighbourTemperature(0, 1 ,0); // +y
    double temperatureBottom = getNeighbourTemperature(0, -1 ,0); // -y
    double temperatureRight = getNeighbourTemperature(1, 0 ,0); // +x
    double temperatureLeft = getNeighbourTemperature(-1, 0 ,0); // -x
    double temperatureForward = getNeighbourTemperature(0, 0 ,1); // +z
    double temperatureBack = getNeighbourTemperature(0, 0, -1); // -z
    double temperatureMiddle = lastFramePhysicalState.temperature;

    double cellSize = grid->cellSideLength();

    //calculating the ∇T (gradient)
    double yGradient = (temperatureTop - temperatureBottom) / (cellSize * 2);
    double xGradient = (temperatureRight - temperatureLeft) / (cellSize * 2);
    double zGradient = (temperatureForward - temperatureBack) / (cellSize * 2);
    dvec3 gradient = dvec3(xGradient, yGradient, zGradient);


    //calculating the ∇^2T (laplace)
    double rateOfChangeOfYGradient = (temperatureTop - temperatureMiddle) - (temperatureMiddle - temperatureBottom);
    rateOfChangeOfYGradient /= pow(cellSize, 2) * 2;
    double rateOfChangeOfZGradient = (temperatureForward - temperatureMiddle) - (temperatureMiddle - temperatureBack) ;
    rateOfChangeOfZGradient /= pow(cellSize, 2) * 2;
    double rateOfChangeOfXGradient =  (temperatureRight - temperatureMiddle) - (temperatureMiddle - temperatureLeft);
    rateOfChangeOfXGradient /= pow(cellSize, 2) * 2;
    double laplace = rateOfChangeOfYGradient + rateOfChangeOfZGradient + rateOfChangeOfXGradient;

    return {gradient, laplace};
}



