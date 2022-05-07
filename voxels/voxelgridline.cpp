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

const int pointsReservedForVoxels = 24;
const int pointsReservedForVectorRendering = 4;
const int pointsReservedForVoxelGridBoundry = 24;

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
    shader->setUniform("renderingGridBoundary", false);

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

    //Drawing the grid boundary as well
    shader->setUniform("m", vec3(0,0,0));
    shader->setUniform("renderingVectorField", false);
    shader->setUniform("renderingGridBoundary", true);
    shader->setUniform("propType", -1);
    glDrawArrays(GL_LINES, pointsReservedForVoxels + pointsReservedForVectorRendering, pointsReservedForVoxelGridBoundry);

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

    //Adding points for the voxels
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
    float scale = 1.3;
    vertices.insert(vertices.end(), {
                        0, 0, 0,
                        0, scale * halfCellLength, 0,
                        0, scale * halfCellLength, 0,
                        -0.3f * halfCellLength, 0.6f * scale * halfCellLength, 0,
                    });

    //Adding points for the boundary
    vec3 gridMin = grid->getMinXYZ();
    int gridSize = grid->getAxisSize();
    vec3 gridMax = gridMin + vec3(gridSize, gridSize, gridSize);

    vec3 offsetsForBoundry [] = {
        vec3(gridMin.x, gridMin.y, gridMin.z), //(0, 0, 0)
        vec3(gridMax.x, gridMin.y, gridMin.z), //(1, 0, 0)
        vec3(gridMax.x, gridMax.y, gridMin.z), //(1, 1, 0)
        vec3(gridMin.x, gridMax.y, gridMin.z), //(0, 1, 0)
        vec3(gridMin.x, gridMin.y, gridMax.z), //(0, 0, 1)
        vec3(gridMax.x, gridMin.y, gridMax.z), //(1, 0, 1)
        vec3(gridMax.x, gridMax.y, gridMax.z), //(1, 1, 1)
        vec3(gridMin.x, gridMax.y, gridMax.z) //(0, 1, 1)
    };

    for (int line = 0; line < 12; line++){
        vertices.insert(vertices.end(), {
                            offsetsForBoundry[(int)lines[line].x].x, offsetsForBoundry[(int)lines[line].x].y, offsetsForBoundry[(int)lines[line].x].z,
                            offsetsForBoundry[(int)lines[line].y].x, offsetsForBoundry[(int)lines[line].y].y, offsetsForBoundry[(int)lines[line].y].z,
                        });
    }

}

void VoxelGridLine::renderVoxel(Voxel *vox, bool renderingInEyeMode){
    if (vox == nullptr) return;
    float temperature = vox->getCurrentState()->temperature;
    //Doing this to sense explosions
    bool isValidTemperature = !std::isnan(temperature) && abs(temperature) < 2000;
    vec3 pos = vec3(vox->centerInWorldSpace);

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
        //Still pretty tightly coupled with temperature ranges, even if not using TEMPERATURE
        if (voxelMode == TEMP_LAPLACE){
            shader->setUniform("prop", (float)vox->getCurrentState()->tempLaplaceFromPrevState);
        }else if (voxelMode == TEMPERATURE){
            shader->setUniform("prop", temperature);
            shader->setUniform("selectedVoxel", !isValidTemperature);
        }else if (voxelMode == WATER){
            shader->setUniform("prop", (float)vox->getCurrentState()->q_v);
            shader->setUniform("secondProp", (float)vox->getCurrentState()->q_c);
        }
        shader->setUniform("renderingVectorField", false);
        glDrawArrays(GL_LINES, 0, pointsReservedForVoxels);
    }

    if (vectorFieldEnabled){
        //Now using the last part of the VAO to render vector field
        if (vectorMode == UFIELD){
            //TODO: this should be based on the iterpolated u field (to get the u field in the center),
            //not the u field the voxel is storing (which is relative to its faces)
            shader->setUniform("u", vec3(vox->getCurrentState()->u));
        }else{
            shader->setUniform("u", vec3(vox->getCurrentState()->tempGradientFromPrevState));
        }
        shader->setUniform("renderingVectorField", true);
        glDrawArrays(GL_LINES, pointsReservedForVoxels, pointsReservedForVectorRendering);
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
     case WATER:
        return "Red -> more water vapor. Green -> more condensed water";
    }
}
