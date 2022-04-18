#version 330 core
in vec3 worldPosition;
out vec4 FragColor;


uniform float temp;
uniform float tempMin;
uniform float tempMax;
uniform bool renderingufield;

void main()
{
   float lerp = min((temp - tempMin) / (tempMax - tempMin), 1);
   float alpha = 0.3 + lerp * 0.7;
   if (renderingufield){
       FragColor = vec4(0, 1, 1, alpha);
   }else{
       FragColor = vec4(1, 1 - lerp, 0, alpha);
   }

}
