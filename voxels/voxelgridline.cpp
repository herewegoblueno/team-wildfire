#include "voxelgridline.h"
#include "GL/glew.h"
#include <iostream>
#include "support/gl/GLDebug.h"
#include "support/lib/ResourceLoader.h"
#include "support/camera/Camera.h"
#include "voxelgrid.h"
#include "support/Settings.h"
#include <math.h>
#include "trees/forest.h"
#include "trees/module.h"


//Modified from https://stackoverflow.com/questions/14486291/how-to-draw-line-in-opengl
void VoxelGridLine::init(VoxelGrid *grid)
{
    this->grid = grid;
    pv = mat4(1.0f);
    forest = nullptr;

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
    if (!voxelsGridEnabled && !vectorFieldEnabled) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    shader.get()->bind();
    shader->setUniform("PV", pv);
    shader->setUniform("propMin", temperatureThreshold);
    shader->setUniform("propMax", temperatureMax);
    shader->setUniform("propType", voxelMode);

    Voxel *eyeVoxel = grid->getVoxelClosestToPoint(eyeCenter);
    int eyeRadiusInVoxels = (int)ceil(eyeRadius / grid->cellSideLength());
    int eyeVoxelX = eyeVoxel->XIndex;
    int eyeVoxelY = eyeVoxel->YIndex;
    int eyeVoxelZ = eyeVoxel->ZIndex;

    glBindVertexArray(VAO);

    if (forest != nullptr && settings.visualizeOnlyVoxelsTouchingSelectedModule && settings.selectedModuleId != DEFAULT_MODULE_ID){
        //Render the branches currently being touched by the current branch
        Module *m = forest->getModuleFromId(settings.selectedModuleId);
        if (m == nullptr) return;
        VoxelSet mappedVoxels = forest->getVoxelsMappedToModule(m);
        for (Voxel *v : mappedVoxels) renderVoxel(v, false);
    }else{
        //Render based off the sliding window...
        for (int x = eyeVoxelX - eyeRadiusInVoxels; x <= eyeVoxelX + eyeRadiusInVoxels; x++){
            for (int y = eyeVoxelY - eyeRadiusInVoxels; y <= eyeVoxelY + eyeRadiusInVoxels; y++){
                for (int z = eyeVoxelZ - eyeRadiusInVoxels; z <= eyeVoxelZ + eyeRadiusInVoxels; z++){
                    renderVoxel(grid->getVoxel(x, y, z), true);
                }
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
    voxelMode = settings.voxelGridMode;
    vectorMode = settings.vectorGridMode;
}

void VoxelGridLine::generateGridVertices(VoxelGrid *grid){

    float halfCellLength = (grid->cellSideLength() / 2.0) * 0.8;
    vec3 offsets [] = {
        vec3(-halfCellLength, -halfCellLength, -halfCellLength), //(0, 0, 0)
        vec3(halfCellLength, -halfCellLength, -halfCellLength), //(1, 0, 0)
        vec3(halfCellLength, halfCellLength, -halfCellLength), //(1, 1, 0)
        vec3(-halfCellLength, halfCellLength, -halfCellLength), //(0, 1, 0)
        vec3(-halfCellLength, -halfCellLength, halfCellLength), //(0, 0, 1)
        vec3(halfCellLength, -halfCellLength, halfCellLength), //(1, 0, 1)
        vec3(halfCellLength, halfCellLength, halfCellLength), //(1, 1, 1)
        vec3(-halfCellLength, halfCellLength, halfCellLength) //(0, 1, 1)
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

    //Add two more points that we'll use for drawing vector fields
    vertices.insert(vertices.end(), { 0, 0, 0, 0, halfCellLength * 5, 0});
}

void VoxelGridLine::renderVoxel(Voxel *vox, bool renderingInEyeMode){
    if (vox == nullptr) return;
    float temperature = vox->getCurrentState()->temperature;
    //Doing this to sense explosions
    bool isValidTemperature = !isnan(temperature) && abs(temperature) < 2000;
    vec3 pos = vox->centerInWorldSpace;

    //We'd love to do this in the shader, but its better to do this here for performance
    if (isValidTemperature){
        if (temperature < temperatureThreshold) return;
        if (temperature > temperatureMax) return;
        if (renderingInEyeMode && glm::length(vec3(pos - eyeCenter)) > eyeRadius) return;
    }else{
        //Uncomment this if you want the simulation to pause the simulation
        //once you start getting bad values....
        //settings.simulatorTimescale = 0;
    }

    shader->setUniform("m", pos);

    if (voxelsGridEnabled){
        //Drawing the cube of the voxel itself...
        //TODO: improve this, still too tightly coupled with temperature ranges
        if (voxelMode == TEMP_LAPLACE){
            shader->setUniform("prop", (float)vox->getCurrentState()->tempLaplaceFromPrevState);
        }else{
            shader->setUniform("selectedVoxel", !isValidTemperature);
            shader->setUniform("prop", temperature);
        }
        shader->setUniform("renderingVectorField", false);
        glDrawArrays(GL_LINES, 0, vertices.size() / 3 - 2);
    }

    if (vectorFieldEnabled){
        //Now using the last part of the VAO to render vector field
        if (vectorMode == UFIELD){
            shader->setUniform("u", vec3(vox->getCurrentState()->u));
        }else{
            shader->setUniform("u", vec3(vox->getCurrentState()->tempGradientFromPrevState));
        }
        shader->setUniform("renderingVectorField", true);
        glDrawArrays(GL_LINES,  (vertices.size() / 3) - 2, 2);
    }
}

void VoxelGridLine::toggle(bool enableVoxels, bool enableWind){
    voxelsGridEnabled = enableVoxels;
    vectorFieldEnabled = enableWind;
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

void VoxelGridLine::setForestReference(Forest *forest){
    this->forest = forest;
}


std::string VoxelGridLine::getVectorFieldModeExplanation(VectorFieldVisualizationModes mode){
    switch (mode){
    case UFIELD:
        return "The lines follow the u-field of the simulation (mostly wind)";
    case TEMP_GRADIENT:
        return "The lines point towards higher temperature. Heat will flow in the opposite direction.";
    }
}

std::string VoxelGridLine::getVoxelFieldModeExplanation(VoxelVisualizationModes mode){
    switch (mode){
    case TEMPERATURE:
        return "Scale: Yellow (low temp) -> Red (high temp). Green = Invalid Temp";
    case TEMP_LAPLACE:
        return "Scale: Cyan (--) -> Blue (-) -> Red (+) -> Magenta (++)";
    }
}
