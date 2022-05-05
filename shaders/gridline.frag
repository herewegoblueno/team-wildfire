 #version 330 core
in vec3 worldPosition;
out vec4 FragColor;


uniform float prop; //the property we'll be using to render the color of the cube
uniform float secondProp; //currently only used when using WATER VoxelVisualizationMode
uniform int propType; //based on VoxelVisualizationModes in voxelgridline.h
uniform float propMin;
uniform float propMax;
uniform bool renderingVectorField;
uniform bool selectedVoxel;
uniform bool renderingGridBoundary;

void main()
{
    float alpha = 0;

    if (propType == 0){ //Coloring based on temperature
       float lerp = min((prop - propMin) / (propMax - propMin), 1);
       alpha = 0.3 + lerp * 0.7;

       if (!renderingVectorField){
           FragColor = vec4(1, 1 - lerp, 0, alpha); //yellow to red
       }

    }else if (propType == 1){ //Coloring based on temperature laplace
        float lerp = abs(prop) / propMax;
        alpha = 0.3 + lerp * 0.7;

        if (!renderingVectorField){
            if (prop > 0) FragColor = vec4(1, 0, lerp, alpha); //red to magenta
        }

    }else if (propType == 2){ //Coloring based on water content
            alpha = 1;

            //qv + qc + qr <= 1
            if (!renderingVectorField){
                FragColor = vec4(prop, secondProp, 0, alpha); //red, yellow, green
            }
        }

    if (renderingGridBoundary) FragColor = vec4(1, 1, 1, 1); //white
    if (renderingVectorField) FragColor = vec4(0, 1, 1, alpha); //cyan
    if (selectedVoxel) FragColor = vec4(0.22, 1, 0.1, 1); //neon green

}
