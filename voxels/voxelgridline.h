#ifndef VOXELGRIDLINE_H
#define VOXELGRIDLINE_H

#include "GL/glew.h"
#include <glm.hpp>
#include <vector>
#include "support/gl/shaders/CS123Shader.h"
#include <memory>
#include "support/scenegraph/SupportCanvas3D.h"


using namespace glm;
using namespace std;

enum VoxelVisualizationModes { TEMPERATURE, TEMP_LAPLACE, WATER, HUMIDITY };
enum VectorFieldVisualizationModes { UFIELD, TEMP_GRADIENT };

class VoxelGrid;
class Voxel;
class Forest;

class VoxelGridLine
{
public:
    ~VoxelGridLine();

    void init(VoxelGrid *grid);

    void setColor(vec4 color);
    void setPV(mat4 pv);
    void draw(SupportCanvas3D *context);
    void toggle(bool enableVoxels, bool enableWind);
    void updateValuesFromSettings();
    vec3 getEyeCenter();
    void setEyeCenter(vec3 v);
    float getEyeRadius();
    void setEyeRadius(float r);
    void setForestReference(Forest *forest);

    static std::string getVectorFieldModeExplanation(VectorFieldVisualizationModes mode);
    static std::string getVoxelFieldModeExplanation(VoxelVisualizationModes mode);

private:
    void generateGridVertices(VoxelGrid *grid);
    void renderVoxel(Voxel *v, bool renderingInEyeMode);
    unique_ptr<CS123::GL::CS123Shader> shader;
    unsigned int VBO, VAO;
    vector<float> vertices;

    mat4 pv;
    vec3 eyeCenter;
    float eyeRadius;
    float temperatureThreshold;
    float temperatureMax;

    VoxelVisualizationModes voxelMode;
    VectorFieldVisualizationModes vectorMode;
    bool voxelsGridEnabled;
    bool vectorFieldEnabled;

    VoxelGrid *grid;
    Forest *forest;
};

#endif // VOXELGRIDLINE_H
