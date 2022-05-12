#version 330 core
layout (location = 0) in vec3 position;
layout (location = 5) in vec2 texCoord; // UV texture coordinates

out vec2 TexCoords;
out float condensation;
out float humidity;
out float x_off;
out float y_off;


uniform mat4 m;
uniform mat4 v;
uniform mat4 p;
uniform float q_c;
uniform float humi;
uniform float scale;
uniform float x_offset;
uniform float y_offset;

void main()
{

    mat3 invViewRot = inverse(mat3(v));
    vec3 pos        = invViewRot * position;
    vec4 position_cameraSpace = v * m * vec4(pos*scale, 1.0);
    TexCoords = texCoord;
    condensation = q_c>0.1 ? q_c : 0.1;
    humidity = (humi-0.5)*2;
    vec4 world_pos = p * position_cameraSpace;
    x_off = world_pos.y*world_pos.y+world_pos.x;
    y_off = world_pos.y*world_pos.y+world_pos.z;
    gl_Position = p * position_cameraSpace;
}
