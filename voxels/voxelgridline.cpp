#include "voxelgridline.h"
#include "GL/glew.h"
#include <iostream>
#include "support/gl/GLDebug.h"
#include "support/lib/ResourceLoader.h"
#include "support/camera/Camera.h"


//Modified from https://stackoverflow.com/questions/14486291/how-to-draw-line-in-opengl
VoxelGridLine::VoxelGridLine(int axisSize, vec3 offset, int resolution)
{
    lineColor = vec4(1,1,1,1);
    MVP = mat4(1.0f);
    generateGridVertices(axisSize, offset, resolution);

    std::string fragmentShaderSource = ResourceLoader::loadResourceFileToString(":/shaders/gridline.frag");
    std::string vertexShaderSource = ResourceLoader::loadResourceFileToString(":/shaders/gridline.vert");

    shader = std::make_unique<CS123::GL::CS123Shader>(vertexShaderSource, fragmentShaderSource);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void VoxelGridLine::setMVP(mat4 mvp) {
    MVP = mvp;
}

void VoxelGridLine::setColor(vec4 c){
    lineColor = c;
}

void VoxelGridLine::draw(SupportCanvas3D *) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    vec4 homogeneousCamPos = inverse(MVP) * vec4(0,0,-1,0);
    vec3 worldCameraPos = (homogeneousCamPos / homogeneousCamPos.w).xyz();
    shader.get()->bind();
    shader->setUniform("MVP", MVP);
    shader->setUniform("color", lineColor);
    shader->setUniform("camPos", worldCameraPos);

    glBindVertexArray(VAO);
    glDrawArrays(GL_LINES, 0, vertices.size() / 3);

    glDisable(GL_BLEND);
}

VoxelGridLine::~VoxelGridLine() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void VoxelGridLine::generateGridVertices(int axisSize, vec3 offset, int resolution){
    vec3 minXYZ = offset - vec3(axisSize, axisSize, axisSize) / 2.f;
    vec3 maxXYZ = offset + vec3(axisSize, axisSize, axisSize) / 2.f;
    float iterOffset = axisSize * (1.f / resolution);
    for (int iteration = 0; iteration < resolution; iteration++){
        for (int subiteration = 0; subiteration < resolution; subiteration++){
            //Plane of lines going up and down (Y) plane extends along Z axis, stepts forward along X with every new iteration
            vertices.insert(vertices.end(), {
                                minXYZ.x + iterOffset * iteration, minXYZ.y, minXYZ.z + iterOffset * subiteration,
                                minXYZ.x + iterOffset * iteration, maxXYZ.y, minXYZ.z + iterOffset * subiteration
                            });

            //Plane of lines going back and foward (Z), plane extends along X axis, stepts forward along Y with every new iteration
            vertices.insert(vertices.end(), {
                                minXYZ.x + iterOffset * subiteration, minXYZ.y + iterOffset * iteration, minXYZ.z,
                                minXYZ.x + iterOffset * subiteration, minXYZ.y + iterOffset * iteration, maxXYZ.z
                            });

            //Plane of lines going back and foward (X), plane extends along Y axis, stepts forward along Z with every new iteration
            vertices.insert(vertices.end(), {
                                minXYZ.x, minXYZ.y + iterOffset * subiteration, minXYZ.z + iterOffset * iteration,
                                maxXYZ.x, minXYZ.y + iterOffset * subiteration, minXYZ.z + iterOffset * iteration
                            });
        }
     }
}
