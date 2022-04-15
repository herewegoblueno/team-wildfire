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

class VoxelGridLine
{
public:
    VoxelGridLine(int axisSize, vec3 offset, int resolution);
    ~VoxelGridLine();

    void setColor(vec4 color);
    void setMVP(mat4 mvp);
    void draw(SupportCanvas3D *context);
    void toggle(bool enabled);

private:
    void generateGridVertices(int axisSize, vec3 offset, int resolution);
    unique_ptr<CS123::GL::CS123Shader> shader;
    unsigned int VBO, VAO;
    vector<float> vertices;
    mat4 MVP;
    vec4 lineColor;

    bool isEnabled = false;
};

#endif // VOXELGRIDLINE_H
