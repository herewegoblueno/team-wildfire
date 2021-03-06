#ifndef VOXELGRID_H
#define VOXELGRID_H

#include <glm.hpp>
#include "GL/glew.h"
#include "voxelgridline.h"
#include "voxel.h"
#include <unordered_set>
#include "simulation/physics.h"

using namespace glm;
using namespace std;

struct VoxelGridDim {
    vec3 center;
    float axisSize;
    VoxelGridDim(vec3 center, float axisSize) :
        center(center),
        axisSize(axisSize)
    {}
};

class Voxel;

typedef std::unordered_set<Voxel *> VoxelSet;

class VoxelGrid
{
public:
    VoxelGrid(VoxelGridDim dimensions, int resolution);
    void toggleVisualization(bool enableVoxels, bool enableWind);
    int getResolution();
    Voxel *getVoxel(int xIndex, int yIndex, int zIndex);
    Voxel *getVoxelClosestToPoint(vec3 point);
    VoxelPhysicalData getStateInterpolatePoint(vec3 point);
    float getQcInterpolatePoint(vec3 point);
    dvec3 getVelInterpolatePoint(vec3 point);
    double getVel(int xIndex, int yIndex, int zIndex, int dim);
    VoxelGridLine *getVisualization();
    double getVolumePerCell();
    double cellSideLength();
    double cellSideLengthForGradients();

    bool isGoodIndex(int i);
    int getClampedIndex(int i);

    void setGlobalFField(vec3 f);
    dvec3 getGlobalFField();

    vec3 getMinXYZ();
    int getAxisSize();

private:
    VoxelGridLine gridlines;
    int resolution;
    int axisSize;
    vec3 minXYZ;
    int overallNumberOfCells;
    vec3 offset;
    //3D array of Voxels...
    //Interstingly, flattening this to 1D is slower (when i tried it, at least)
    vector<vector<vector<std::unique_ptr<Voxel>>>> voxels;
    float cellVolume;

    dvec3 globalFField; //Global wind (Stormscapes paper Eq 20), think of it like global wind
};



#endif // VOXELGRID_H
