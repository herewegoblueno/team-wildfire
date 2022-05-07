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

//When calculating gradients, we act as if the cells are larger than they actually are for better
//simulation stability
double VoxelGrid::cellSideLengthForGradients(){
    return cellSideLength() * voxelSizeMultiplierForGradients;
}

Voxel *VoxelGrid::getVoxel(int xIndex, int yIndex, int zIndex){
    if (zIndex >= resolution || yIndex >= resolution || xIndex >= resolution) return nullptr;
    if (zIndex < 0 || yIndex < 0 || xIndex < 0) return nullptr;
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

    VoxelPhysicalData output;
    Voxel* p000 = getVoxel(xIndex, yIndex, zIndex);
    Voxel* p001 = getVoxel(xIndex, yIndex, zIndex+1);
    Voxel* p010 = getVoxel(xIndex, yIndex+1, zIndex);
    Voxel* p011 = getVoxel(xIndex, yIndex+1, zIndex+1);
    Voxel* p100 = getVoxel(xIndex+1, yIndex, zIndex);
    Voxel* p101 = getVoxel(xIndex+1, yIndex, zIndex+1);
    Voxel* p110 = getVoxel(xIndex+1, yIndex+1, zIndex);
    Voxel* p111 = getVoxel(xIndex+1, yIndex+1, zIndex+1);
    if(p000==nullptr || p001==nullptr || p010==nullptr || p011==nullptr ||
       p100==nullptr || p101==nullptr || p110==nullptr || p111==nullptr)
    {
        if(getVoxel(xIndex, yIndex, zIndex) != nullptr)
            return *getVoxel(xIndex, yIndex, zIndex)->getCurrentState();
        else
            return VoxelPhysicalData();
    }


    double xd = (point.x - p000->centerInWorldSpace.x - voxelSize*0.5)/voxelSize;
    double yd = (point.y - p000->centerInWorldSpace.y - voxelSize*0.5)/voxelSize;
    double zd = (point.z - p000->centerInWorldSpace.z - voxelSize*0.5)/voxelSize;

    VoxelPhysicalData c00 = (*p000->getCurrentState())*(1.-xd) + (*p100->getCurrentState())*xd;
    VoxelPhysicalData c01 = (*p001->getCurrentState())*(1.-xd) + (*p101->getCurrentState())*xd;
    VoxelPhysicalData c10 = (*p010->getCurrentState())*(1.-xd) + (*p110->getCurrentState())*xd;
    VoxelPhysicalData c11 = (*p011->getCurrentState())*(1.-xd) + (*p111->getCurrentState())*xd;
    VoxelPhysicalData c0 = c00*(1.-yd) + c10*yd;
    VoxelPhysicalData c1 = c01*(1.-yd) + c11*yd;

    VoxelPhysicalData out = c0*(1-zd) + c1*zd;
    out.u.x = (1-xd)*getVel(xIndex-1, yIndex, zIndex, 0) + xd*getVel(xIndex, yIndex, zIndex, 0);
    out.u.y = (1-yd)*getVel(xIndex, yIndex-1, zIndex, 1) + yd*getVel(xIndex, yIndex, zIndex, 1);
    out.u.z = (1-zd)*getVel(xIndex, yIndex, zIndex-1, 2) + zd*getVel(xIndex, yIndex, zIndex, 2);
    return out;
}

//Not used at the moment, added in case it becomes useful
dvec3 VoxelGrid::getVelInterpolatePoint(vec3 point)
{
    vec3 distancesToPoint = point - minXYZ;
    float voxelSize = cellSideLength();
    int xIndex = static_cast<int>(clamp(floor(distancesToPoint.x / voxelSize), 0.f, float(resolution - 1)));
    int yIndex = static_cast<int>(clamp(floor(distancesToPoint.y / voxelSize), 0.f, float(resolution - 1)));
    int zIndex = static_cast<int>(clamp(floor(distancesToPoint.z / voxelSize), 0.f, float(resolution - 1)));
    Voxel* p000 = getVoxel(xIndex, yIndex, zIndex);
    double xd = (point.x - p000->centerInWorldSpace.x)/voxelSize;
    double yd = (point.y - p000->centerInWorldSpace.y)/voxelSize;
    double zd = (point.z - p000->centerInWorldSpace.z)/voxelSize;
    dvec3 u;
    u.x = (1-xd)*getVel(xIndex-1, yIndex, zIndex, 0) + xd*getVel(xIndex, yIndex, zIndex, 0);
    u.y = (1-yd)*getVel(xIndex, yIndex-1, zIndex, 1) + yd*getVel(xIndex, yIndex, zIndex, 1);
    u.z = (1-zd)*getVel(xIndex, yIndex, zIndex-1, 2) + zd*getVel(xIndex, yIndex, zIndex, 2);
    return u;
}


double VoxelGrid::getVolumePerCell(){
    return cellVolume;
}

vec3 VoxelGrid::getMinXYZ(){
    return minXYZ;
}

int VoxelGrid::getAxisSize(){
    return axisSize;
}

bool VoxelGrid::isGoodIndex(int i){
    return i >= 0 && i < resolution;
}

int VoxelGrid::getClampedIndex(int i){
    return clamp(i, 0, resolution - 1);
}

dvec3 VoxelGrid::getGlobalFField(){
    return globalFField;
}

void VoxelGrid::setGlobalFField(vec3 f){
    globalFField = f;

    //If we're going through the whole physics simulation (which makes use of CUDA to edit the u
    //field of voxels for things like vortexes), then we shouldn't alter the u field of voxels directly.
    //(f field is taken into account there).
    //If we aren't, then the u field is never actually altered during the simulation, so we should just alter
    //it here.
#ifdef CUDA_FLUID
    return;
#endif

    for (int x = 0; x < resolution; x++){
        for (int y = 0; y < resolution; y++){
            for (int z = 0; z < resolution; z++){
                Voxel *v = getVoxel(x, y, z);
                v->getCurrentState()->u = f;
                v->getLastFrameState()->u = f;
            }
        }
    }
}


double VoxelGrid::getVel(int x, int y, int z, int dim)
{
    Voxel* v = getVoxel(x, y, z);
    if (v == nullptr) return 0;
    int resolution = getResolution();
    dvec3 u = v->getCurrentState()->u;
    if(dim == 0) {
        if(x == resolution - 1) return 0;
        else return u.x;
    } else if(dim == 1) {
        if(y == resolution - 1) return 0;
        else return u.y;
    } else if(dim == 2) {
        if(z == resolution - 1) return 0;
        else return u.z;
    } return 0;
}








