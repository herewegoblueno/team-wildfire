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
    lastFramePhysicalState.temperature = getAmbientTempAtIndices(XIndex, YIndex, ZIndex);
    currentPhysicalState.temperature = getAmbientTempAtIndices(XIndex, YIndex, ZIndex);

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


double Voxel::getAmbientTempAtIndices(int, int y, int){
    y = glm::clamp(y, 0, grid->getResolution() - 1);
    return glm::clamp(3.5 - y * grid->cellSideLength() / 2, 0.0, 3.5 );
}

double Voxel::getAmbientTemperature(){
    return getAmbientTempAtIndices(XIndex, YIndex, ZIndex);
}


double Voxel::getNeighbourTemperature(int xOffset, int yOffset, int zOffset){
    Voxel *vox = getVoxelWithIndexOffset({xOffset, yOffset, zOffset});
     double temp = vox == nullptr ? getAmbientTempAtIndices(xOffset + XIndex, yOffset + YIndex, zOffset + ZIndex) : vox->getLastFrameState()->temperature;
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


dvec3 Voxel::getGradient(double (*func)(Voxel *))
{
    Voxel *vox = getVoxelWithIndexOffset(vec3(0,1,0));
    double temperatureTop = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,-1,0));
    double temperatureBottom = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(1,0,0));
    double temperatureRight = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(-1,0,0));
    double temperatureLeft = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,0,1));
    double temperatureForward = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,0,-1));
    double temperatureBack = vox == nullptr ? 0 : func(vox);

    float cellSize = grid->cellSideLength();

    //calculating the ∇T (gradient)
    double yGradient = (temperatureTop - temperatureBottom) / (cellSize * 2);
    double xGradient = (temperatureRight - temperatureLeft) / (cellSize * 2);
    double zGradient = (temperatureForward - temperatureBack) / (cellSize * 2);
    dvec3 gradient = dvec3(xGradient, yGradient, zGradient);
    return gradient;
}


double Voxel::getLaplace(double (*func)(Voxel *))
{
    Voxel *vox = getVoxelWithIndexOffset(vec3(0,1,0));
    double temperatureTop = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,-1,0));
    double temperatureBottom = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(1,0,0));
    double temperatureRight = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(-1,0,0));
    double temperatureLeft = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,0,1));
    double temperatureForward = vox == nullptr ? 0 : func(vox);
    vox = getVoxelWithIndexOffset(vec3(0,0,-1));
    double temperatureBack = vox == nullptr ? 0 : func(vox);
    double temperatureMiddle = func(this);

    float cellSize = grid->cellSideLength();

    double rateOfChangeOfYGradient = (temperatureTop - temperatureMiddle) - (temperatureMiddle - temperatureBottom);
    rateOfChangeOfYGradient /= pow(cellSize, 2) * 2;
    double rateOfChangeOfZGradient = (temperatureForward - temperatureMiddle) - (temperatureMiddle - temperatureBack) ;
    rateOfChangeOfZGradient /= pow(cellSize, 2) * 2;
    double rateOfChangeOfXGradient =  (temperatureRight - temperatureMiddle) - (temperatureMiddle - temperatureLeft);
    rateOfChangeOfXGradient /= pow(cellSize, 2) * 2;
    double laplace = rateOfChangeOfYGradient + rateOfChangeOfZGradient + rateOfChangeOfXGradient;
    return laplace;
}


dvec3 Voxel::getVerticity()
{
    dvec3 grad_ux = getGradient(get_q_ux);
    dvec3 grad_uy = getGradient(get_q_uy);
    dvec3 grad_uz = getGradient(get_q_uz);

    return dvec3(grad_uz.y - grad_uy.z, grad_ux.z - grad_uz.x, grad_uy.x - grad_ux.y);
}


double get_q_ux(Voxel* v) {return v->getLastFrameState()->u.x;}
double get_q_uy(Voxel* v) {return v->getLastFrameState()->u.y;}
double get_q_uz(Voxel* v) {return v->getLastFrameState()->u.z;}

double get_q_v(Voxel* v) {return v->getLastFrameState()->q_v;}
double get_q_c(Voxel* v) {return v->getLastFrameState()->q_c;}
double get_q_r(Voxel* v) {return v->getLastFrameState()->q_r;}
