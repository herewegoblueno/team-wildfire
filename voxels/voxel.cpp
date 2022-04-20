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
vec3 Voxel::getTemperatureGradientFromPreviousFrame(){
    float averageTemperatureTop = 0; // +y
    float averageTemperatureBottom = 0; // -y
    float averageTemperatureLeft = 0; // -x
    float averageTemperatureRight = 0; // +x
    float averageTemperatureBack = 0; // -z
    float averageTemperatureForward = 0; // +z

    for (int x = 0; x < 3; x++){
        for (int y = 0; y < 3; y++){
            for (int z = 0; z < 3; z++){
                vec3 offset(x - 1, y - 1, z - 1);
                Voxel *vox = getVoxelWithIndexOffset(offset);
                //TODO: this should actually be the getAmbientTemperature of the center of this out-of-bounds voxel, not this voxel
                float temp = vox == nullptr ? getAmbientTemperature(centerInWorldSpace) : vox->getLastFrameState()->temperature;
                if (x == 0) averageTemperatureLeft += temp / 9.f;
                if (x == 2) averageTemperatureRight += temp / 9.f;
                if (y == 0) averageTemperatureBottom += temp / 9.f;
                if (y == 2) averageTemperatureTop += temp / 9.f;
                if (z == 0) averageTemperatureBack += temp / 9.f;
                if (z == 2) averageTemperatureForward += temp / 9.f;
            }
        }
    }

    float cellSize = grid->cellSideLength();
    float yGradient = (averageTemperatureTop - averageTemperatureBottom) / (cellSize * 2);
    float xGradient = (averageTemperatureRight - averageTemperatureLeft) / (cellSize * 2);
    float zGradient = (averageTemperatureForward - averageTemperatureBack) / (cellSize * 2);
    return vec3(xGradient, yGradient, zGradient);
}



