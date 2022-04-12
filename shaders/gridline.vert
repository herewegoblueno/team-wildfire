#version 330 core

 layout (location = 0) in vec3 aPos;
 out vec3 worldPosition;

 uniform mat4 MVP;
 void main()
 {
    worldPosition = aPos;
    gl_Position = MVP * vec4(aPos.x, aPos.y, aPos.z, 1.0);
 }
