#include "voxelgridline.h"
#include "GL/glew.h"
#include <iostream>
#include "support/gl/GLDebug.h"
#include "support/lib/ResourceLoader.h"
#include "support/camera/Camera.h"
#include "voxelgrid.h"
#include "support/Settings.h"
#include <math.h>


//Modified from https://stackoverflow.com/questions/14486291/how-to-draw-line-in-opengl
void VoxelGridLine::init(VoxelGrid *grid)
{
    this->grid = grid;
    pv = mat4(1.0f);
    updateValuesFromSettings();

    generateGridVertices(grid);

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

void VoxelGridLine::setPV(mat4 pv) {
    this->pv = pv;
}

void VoxelGridLine::draw(SupportCanvas3D *) {
    if (!isEnabled) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    shader.get()->bind();
    shader->setUniform("PV", pv);
    shader->setUniform("tempMin", temperatureThreshold);
    shader->setUniform("tempMax", temperatureMax);

    Voxel *eyeVoxel = grid->getVoxelClosestToPoint(eyeCenter);
    int eyeRadiusInVoxels = (int)ceil(eyeRadius / grid->cellSideLength());
    int eyeVoxelX = eyeVoxel->XIndex;
    int eyeVoxelY = eyeVoxel->YIndex;
    int eyeVoxelZ = eyeVoxel->ZIndex;

    glBindVertexArray(VAO);
    for (int x = eyeVoxelX - eyeRadiusInVoxels; x <= eyeVoxelX + eyeRadiusInVoxels; x++){
        for (int y = eyeVoxelY - eyeRadiusInVoxels; y <= eyeVoxelY + eyeRadiusInVoxels; y++){
            for (int z = eyeVoxelZ - eyeRadiusInVoxels; z <= eyeVoxelZ + eyeRadiusInVoxels; z++){
                if (grid->getVoxel(x, y, z) == nullptr) continue;
                float temperature = grid->getVoxel(x, y, z)->getCurrentState()->temperature;
                vec3 pos = grid->getVoxel(x, y, z)->centerInWorldSpace;

                //We'd love to do this in the shader, but its better to do this here for performance
                if (temperature < temperatureThreshold) continue;
                if (glm::length(vec3(pos - eyeCenter)) > eyeRadius) continue;

                shader->setUniform("m", grid->getVoxel(x, y, z)->centerInWorldSpace);
                shader->setUniform("temp", temperature);
                glDrawArrays(GL_LINES, 0, vertices.size() / 3);
            }
        }
    }


    glDisable(GL_BLEND);
}

VoxelGridLine::~VoxelGridLine() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

void VoxelGridLine::updateValuesFromSettings(){
    temperatureThreshold = settings.visualizeForestVoxelGridMinTemp;
    temperatureMax = settings.visualizeForestVoxelGridMaxTemp;
    eyeCenter = vec3(
                settings.visualizeForestVoxelGridEyeX,
                settings.visualizeForestVoxelGridEyeY,
                settings.visualizeForestVoxelGridEyeZ);
    eyeRadius = settings.visualizeForestVoxelGridEyeRadius;
}

void VoxelGridLine::generateGridVertices(VoxelGrid *grid){

    float halfCellLength = (grid->cellSideLength() / 2.0) * 0.8;
    vec3 offsets [] = {
        vec3(-halfCellLength, -halfCellLength, -halfCellLength), //(0, 0, 0) 0
        vec3(halfCellLength, -halfCellLength, -halfCellLength), //(1, 0, 0) 1
        vec3(halfCellLength, halfCellLength, -halfCellLength), //(1, 1, 0) 2
        vec3(-halfCellLength, halfCellLength, -halfCellLength), //(0, 1, 0) 3
        vec3(-halfCellLength, -halfCellLength, halfCellLength), //(0, 0, 1) 4
        vec3(halfCellLength, -halfCellLength, halfCellLength), //(1, 0, 1) 5
        vec3(halfCellLength, halfCellLength, halfCellLength), //(1, 1, 1) 6
        vec3(-halfCellLength, halfCellLength, halfCellLength) //(0, 1, 1) 7
    };

    vec2 lines [] = {
        vec2(4, 7),
        vec2(6, 7),
        vec2(6, 5),
        vec2(5, 4),
        vec2(4, 0),
        vec2(5, 1),
        vec2(6, 2),
        vec2(7, 3),
        vec2(0, 1),
        vec2(1, 2),
        vec2(2, 3),
        vec2(3, 0),
    };


    for (int line = 0; line < 12; line++){
        vertices.insert(vertices.end(), {
                            offsets[(int)lines[line].x].x, offsets[(int)lines[line].x].y, offsets[(int)lines[line].x].z,
                            offsets[(int)lines[line].y].x, offsets[(int)lines[line].y].y, offsets[(int)lines[line].y].z,
                        });

    }
}

void VoxelGridLine::toggle(bool enabled){
    isEnabled = enabled;
}

vec3 VoxelGridLine::getEyeCenter(){
    return eyeCenter;
}

void VoxelGridLine::setEyeCenter(vec3 v){
    eyeCenter = v;
}

float VoxelGridLine::getEyeRadius(){
    return eyeRadius;
}

void VoxelGridLine::setEyeRadius(float r){
    eyeRadius = r;
}
