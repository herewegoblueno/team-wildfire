#version 330 core

in vec3 color;
in vec2 texc;
out vec4 fragColor;

uniform float prop; //the property we'll be using to render the color of the cube
uniform int propType; //based on VoxelVisualizationModes in voxelgridline.h
uniform float propMin;
uniform float propMax;

uniform bool isSelected;
uniform bool warningFlag;


void main(){
    if (isSelected){
        //Neon Green
        fragColor = vec4(0.22, 1, 0.1, 1);
    } else if (warningFlag){
        //Bright red
        fragColor = vec4(1.0, 0.1, 0.1, 1);
    }else if (propType == 0){
        //use the color provided by the material (which should correlate to module ID)
        fragColor = vec4(color, 1);
    }else if (propType == 1){
        //Temperature visualization, yellow to red
        float lerp = min((prop - propMin) / (propMax - propMin), 1);
        fragColor = vec4(1, 1 - lerp, 0, 1);
    }
}
