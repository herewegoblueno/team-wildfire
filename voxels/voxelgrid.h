#ifndef VOXELGRID_H
#define VOXELGRID_H

#include <glm.hpp>
#include "GL/glew.h"
#include "voxelgridline.h"
#include "voxel.h"

using namespace glm;
using namespace std;

class Voxel;

class VoxelGrid
{
public:
    VoxelGrid(int axisSize, vec3 offset, int resolution);
    void toggleVisualization(bool enableVoxels, bool enableWind);
    int getResolution();
    Voxel *getVoxel(int xIndex, int yIndex, int zIndex);
    Voxel *getVoxelClosestToPoint(vec3 point);
    VoxelGridLine *getVisualization();
    float getVolumePerCell();
    float cellSideLength();

private:
    VoxelGridLine gridlines;
    int resolution;
    int axisSize;
    vec3 minXYZ;
    int overallNumberOfCells;
    vec3 offset;
    vector<vector<vector<std::unique_ptr<Voxel>>>> voxels; //3D array of Voxels...
    float cellVolume;
};

#endif // VOXELGRID_H
