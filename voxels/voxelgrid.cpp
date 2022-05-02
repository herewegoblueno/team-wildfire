#include "voxelgrid.h"

VoxelGrid::VoxelGrid(int axisSize, vec3 offset, int resolution) :
    resolution(resolution),
    axisSize(axisSize),
    offset(offset)
{
    overallNumberOfCells = std::pow(resolution, 3);
    double overallVoxelSpaceVolume = std::pow(axisSize, 3);
    cellVolume = overallVoxelSpaceVolume / overallNumberOfCells;

    //Making the grid....
    minXYZ = offset - vec3(axisSize, axisSize, axisSize) / 2.f;
    double voxelAxisSize = cellSideLength();

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

    gridlines.init(this);
}

void VoxelGrid::toggleVisualization(bool enableVoxels, bool enableWind){
    gridlines.toggle(enableVoxels, enableWind);
}

VoxelGridLine *VoxelGrid::getVisualization(){
    return &gridlines;
}

int VoxelGrid::getResolution(){
    return resolution;
}

double VoxelGrid::cellSideLength(){
    return axisSize * (1.0 / resolution);
}

Voxel *VoxelGrid::getVoxel(int xIndex, int yIndex, int zIndex){
    if (zIndex >= resolution || yIndex >= resolution || xIndex >= resolution) return nullptr;
    if (zIndex < 0 || yIndex < 0 || xIndex < 0) return nullptr;
    //TODO: hopefully this is not copying allhe intermediate vectors as its accessing the voxel
    return voxels[xIndex][yIndex][zIndex].get();
}

Voxel *VoxelGrid::getVoxelClosestToPoint(vec3 point){
    vec3 distancesToPoint = point - minXYZ;
    float voxelSize = cellSideLength();
    int xIndex = static_cast<int>(clamp(floor(distancesToPoint.x / voxelSize), 0.f, float(resolution - 1)));
    int yIndex = static_cast<int>(clamp(floor(distancesToPoint.y / voxelSize), 0.f, float(resolution - 1)));
    int zIndex = static_cast<int>(clamp(floor(distancesToPoint.z / voxelSize), 0.f, float(resolution - 1)));
    return getVoxel(xIndex, yIndex, zIndex);
}

VoxelPhysicalData VoxelGrid::getStateInterpolatePoint(vec3 point){
    vec3 distancesToPoint = point - minXYZ;
    float voxelSize = cellSideLength();
    int xIndex = static_cast<int>(clamp(floor(distancesToPoint.x / voxelSize), 0.f, float(resolution - 1)));
    int yIndex = static_cast<int>(clamp(floor(distancesToPoint.y / voxelSize), 0.f, float(resolution - 1)));
    int zIndex = static_cast<int>(clamp(floor(distancesToPoint.z / voxelSize), 0.f, float(resolution - 1)));

    if(point.x<minXYZ.x || point.y<minXYZ.y || point.z<minXYZ.z ||
       point.x>minXYZ.x+axisSize || point.y>minXYZ.y+axisSize || point.z>minXYZ.z+axisSize   )
        return *getVoxel(xIndex, yIndex, zIndex)->getCurrentState();

    VoxelPhysicalData output;
    Voxel* p000 = getVoxel(xIndex, yIndex, zIndex);
    Voxel* p001 = getVoxel(xIndex, yIndex, zIndex+1);
    Voxel* p010 = getVoxel(xIndex, yIndex+1, zIndex);
    Voxel* p011 = getVoxel(xIndex, yIndex+1, zIndex+1);
    Voxel* p100 = getVoxel(xIndex+1, yIndex, zIndex);
    Voxel* p101 = getVoxel(xIndex+1, yIndex, zIndex+1);
    Voxel* p110 = getVoxel(xIndex+1, yIndex+1, zIndex);
    Voxel* p111 = getVoxel(xIndex+1, yIndex+1, zIndex+1);

    double xd = (point.x - p000->centerInWorldSpace.x)/voxelSize;
    double yd = (point.y - p000->centerInWorldSpace.y)/voxelSize;
    double zd = (point.z - p000->centerInWorldSpace.z)/voxelSize;

    VoxelPhysicalData c00 = (*p000->getCurrentState())*(1.-xd) + (*p100->getCurrentState())*xd;
    VoxelPhysicalData c01 = (*p001->getCurrentState())*(1.-xd) + (*p101->getCurrentState())*xd;
    VoxelPhysicalData c10 = (*p010->getCurrentState())*(1.-xd) + (*p110->getCurrentState())*xd;
    VoxelPhysicalData c11 = (*p011->getCurrentState())*(1.-xd) + (*p111->getCurrentState())*xd;
    VoxelPhysicalData c0 = c00*(1.-yd) + c10*yd;
    VoxelPhysicalData c1 = c01*(1.-yd) + c11*yd;

    return c0*(1-zd) + c1*zd;
}

double VoxelGrid::getVolumePerCell(){
    return cellVolume;
}


bool VoxelGrid::isGoodIndex(int i){
    return i >= 0 && i < resolution;
}

int VoxelGrid::getClampedIndex(int i){
    return clamp(i, 0, resolution - 1);
}

