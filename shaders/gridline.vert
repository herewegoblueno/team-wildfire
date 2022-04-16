#version 330 core

 layout (location = 0) in vec3 aPos;
 out vec3 worldPosition;

 uniform mat4 PV;
 uniform vec3 m;

 void main()
 {
    worldPosition = aPos + m;
    gl_Position =  PV * vec4(worldPosition, 1.0);
 }
