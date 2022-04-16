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

class VoxelGrid;

class VoxelGridLine
{
public:
    ~VoxelGridLine();

    void init(VoxelGrid *grid);

    void setColor(vec4 color);
    void setPV(mat4 pv);
    void draw(SupportCanvas3D *context);
    void toggle(bool enabled);
    void updateValuesFromSettings();
    vec3 getEyeCenter();
    void setEyeCenter(vec3 v);
    float getEyeRadius();
    void setEyeRadius(float r);

private:
    void generateGridVertices(VoxelGrid *grid);
    unique_ptr<CS123::GL::CS123Shader> shader;
    unsigned int VBO, VAO;
    vector<float> vertices;

    mat4 pv;
    vec3 eyeCenter;
    float eyeRadius;
    float temperatureThreshold;
    float temperatureMax;

    bool isEnabled = false;

    VoxelGrid *grid;
};

#endif // VOXELGRIDLINE_H
