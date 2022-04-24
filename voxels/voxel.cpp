#include "voxel.h"

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
//    int targetIndex = ceil(grid->getResolution() / 2.f);
//    if (XIndex == targetIndex && YIndex == targetIndex && ZIndex == targetIndex){
//        lastFramePhysicalState.temperature = 10;
//    }
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

void Voxel::switchStates(){
    auto prevState = lastFramePhysicalState;
    lastFramePhysicalState = currentPhysicalState;
    currentPhysicalState = prevState;
}


float Voxel::getAmbientTemperature(vec3 pos){
    return glm::clamp(2 - pos.y / 4.f, 0.f, 2.f );
}

//TODO: improve this accuracy and performance could probably be better
 VoxelTemperatureGradientInfo Voxel::getTemperatureGradientInfoFromPreviousFrame(){
    float averageTemperatureTop = 0; // +y
    float averageTemperatureMiddleY = 0; // y
    float averageTemperatureBottom = 0; // -y

    float averageTemperatureRight = 0; // +x
    float averageTemperatureMiddleX = 0; // x
    float averageTemperatureLeft = 0; // -x

    float averageTemperatureForward = 0; // +z
    float averageTemperatureMiddleZ = 0; // x
    float averageTemperatureBack = 0; // -z

    for (int x = 0; x < 3; x++){
        for (int y = 0; y < 3; y++){
            for (int z = 0; z < 3; z++){
                vec3 offset(x - 1, y - 1, z - 1);
                Voxel *vox = getVoxelWithIndexOffset(offset);
                //TODO: this should actually be the getAmbientTemperature of the center of this out-of-bounds voxel, not this voxel
                float temp = vox == nullptr ? 0 : vox->getLastFrameState()->temperature;
                if (x == 0) averageTemperatureLeft += temp / 9.f;
                if (x == 1) averageTemperatureMiddleX += temp / 9.f;
                if (x == 2) averageTemperatureRight += temp / 9.f;

                if (y == 0) averageTemperatureBottom += temp / 9.f;
                if (y == 1) averageTemperatureMiddleY += temp / 9.f;
                if (y == 2) averageTemperatureTop += temp / 9.f;

                if (z == 0) averageTemperatureBack += temp / 9.f;
                if (z == 1) averageTemperatureMiddleZ += temp / 9.f;
                if (z == 2) averageTemperatureForward += temp / 9.f;
            }
        }
    }

    float cellSize = grid->cellSideLength();

    //calculating the ∇T (gradient)
    float yGradient = (averageTemperatureTop - averageTemperatureBottom) / (cellSize * 2);
    float xGradient = (averageTemperatureRight - averageTemperatureLeft) / (cellSize * 2);
    float zGradient = (averageTemperatureForward - averageTemperatureBack) / (cellSize * 2);
    vec3 gradient = vec3(xGradient, yGradient, zGradient);

    //calculating the ∇^2T (laplace)
    float rateOfChangeOfYGradient = (averageTemperatureMiddleY - averageTemperatureBottom) - (averageTemperatureMiddleY - averageTemperatureTop) / (cellSize * 2);
    float rateOfChangeOfZGradient = (averageTemperatureMiddleZ - averageTemperatureBack) - (averageTemperatureMiddleZ - averageTemperatureForward) / (cellSize * 2);
    float rateOfChangeOfXGradient = (averageTemperatureMiddleX - averageTemperatureLeft) - (averageTemperatureMiddleX - averageTemperatureRight) / (cellSize * 2);
    float laplace = rateOfChangeOfYGradient + rateOfChangeOfZGradient + rateOfChangeOfXGradient;

    return {gradient, laplace};
}


vec3 Voxel::getGradient(float (*func)(Voxel *))
{
    float avg_Top = 0, avg_MiddleY = 0, avg_Bottom = 0;
    float avg_Right = 0, avg_MiddleX = 0, avg_Left = 0;
    float avg_Forward = 0, avg_MiddleZ = 0, avg_Back = 0;

    for (int x = 0; x < 3; x++){
        for (int y = 0; y < 3; y++){
            for (int z = 0; z < 3; z++){
                vec3 offset(x - 1, y - 1, z - 1);
                Voxel *vox = getVoxelWithIndexOffset(offset);
                float value = vox == nullptr ? 0 : func(vox);
                if (x == 0) avg_Left += value / 9.f;
                if (x == 1) avg_MiddleX += value / 9.f;
                if (x == 2) avg_Right += value / 9.f;

                if (y == 0) avg_Bottom += value / 9.f;
                if (y == 1) avg_MiddleY += value / 9.f;
                if (y == 2) avg_Top += value / 9.f;

                if (z == 0) avg_Back += value / 9.f;
                if (z == 1) avg_MiddleZ += value / 9.f;
                if (z == 2) avg_Forward += value / 9.f;
            }
        }
    }
    float cellSize = grid->cellSideLength();

    //calculating the ∇T (gradient)
    float yGradient = (avg_Top - avg_Bottom) / (cellSize * 2);
    float xGradient = (avg_Right - avg_Left) / (cellSize * 2);
    float zGradient = (avg_Forward - avg_Back) / (cellSize * 2);
    vec3 gradient = vec3(xGradient, yGradient, zGradient);
    return gradient;
}


