#version 330 core
in vec3 worldPosition;
out vec4 FragColor;


uniform float temp;
uniform float tempMin;
uniform float tempMax;

void main()
{
   float lerp = min((temp - tempMin) / (tempMax - tempMin), 1);
   FragColor = vec4(1, 1 - lerp, 0, 0.3 + lerp * 0.7);
}
