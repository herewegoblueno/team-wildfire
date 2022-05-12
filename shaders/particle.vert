#version 330 core
layout (location = 0) in vec3 position;
layout (location = 5) in vec2 texCoord; // UV texture coordinates

out vec2 TexCoords;
out float Temperature;


uniform mat4 m;
uniform mat4 v;
uniform mat4 p;
uniform float temp;
uniform float scale;

void main()
{

//    float scale = 0.03f;
//    float scale = 0.06f;
    mat3 invViewRot = inverse(mat3(v));
    vec3 pos        = invViewRot * position;
    vec4 position_cameraSpace = v * m * vec4(pos*scale, 1.0);
    TexCoords = texCoord;
    Temperature = temp;
    gl_Position = p * position_cameraSpace;
}
