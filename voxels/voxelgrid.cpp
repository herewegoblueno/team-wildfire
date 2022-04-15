#include "voxelgrid.h"

VoxelGrid::VoxelGrid(int axisSize, vec3 offset, int resolution) :
    gridlines(axisSize, offset, resolution),
    resolution(resolution),
    axisSize(axisSize),
    offset(offset)
{
    overallNumberOfCells = std::pow(resolution, 3);
    float overallVoxelSpaceVolume = std::pow(axisSize, 3);
    cellVolume = overallVoxelSpaceVolume / overallNumberOfCells;

    //Making the grid....
    minXYZ = offset - vec3(axisSize, axisSize, axisSize) / 2.f;
    float voxelAxisSize = axisSize * (1.f / resolution);

    voxels.resize(resolution);
    for (int x = 0; x < resolution; x++){
        voxels[x].resize(resolution);
        for (int y = 0; y < resolution; y++){
            voxels[x][y].resize(resolution);
            for (int z = 0; z < resolution; z++){
                vec3 bottomLeftCorner = minXYZ + vec3{voxelAxisSize * x, voxelAxisSize * y, voxelAxisSize * z};
                vec3 center = bottomLeftCorner + vec3(voxelAxisSize / 2.0);
                voxels[x][y][z] = make_unique<Voxel>(Voxel(this, x, y, z, center));
            }
        }
    }
}

void VoxelGrid::toggleVisualization(bool enabled){
    gridlines.toggle(enabled);
}

VoxelGridLine *VoxelGrid::getVisualization(){
    return &gridlines;
}

int VoxelGrid::getResolution(){
    return resolution;
}

Voxel *VoxelGrid::getVoxel(int xIndex, int yIndex, int zIndex){
    if (zIndex >= resolution || yIndex >= resolution || xIndex >= resolution) return nullptr;
    if (zIndex < 0 || yIndex < 0 || xIndex < 0) return nullptr;
    //TODO: hopefully this is not copying allhe intermediate vectors as its accessing the voxel
    return voxels[xIndex][yIndex][zIndex].get();
}

Voxel *VoxelGrid::getVoxelClosestToPoint(vec3 point){
    vec3 distancesToPoint = point - minXYZ;
    int xIndex = static_cast<int>(clamp(floor(distancesToPoint.x / resolution), 0.f, float(resolution - 1)));
    int yIndex = static_cast<int>(clamp(floor(distancesToPoint.y / resolution), 0.f, float(resolution - 1)));
    int zIndex = static_cast<int>(clamp(floor(distancesToPoint.z / resolution), 0.f, float(resolution - 1)));
    return getVoxel(xIndex, yIndex, zIndex);
}

float VoxelGrid::getVolumePerCell(){
    return cellVolume;
}
