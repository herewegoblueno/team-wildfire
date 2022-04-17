#version 330 core
in vec2 TexCoords;
in vec4 ParticleColor;
in float Temperature;
out vec4 color;

uniform sampler2D sprite;

void main()
{
    vec2 temp = TexCoords - vec2(0.5);
    float f = dot(temp, temp);

    if(Temperature>3.5) color = vec4(0.7, 0.4, 0.3, 0);
    else if(Temperature>3) color = vec4(0.8, 0.4, 0.3, 0);
    else if(Temperature>2) color = vec4(0.8, 0.4, 0.2, 0);
    else if(Temperature>1) color = vec4(0.9, 0.1, 0.0, 0);
    else color = vec4(0.8, 0, 0, 0);

//    color = (texture(sprite, TexCoords)+0.2)*color;
//    color = ParticleColor;
//    color.a = (color.x+color.y+color.z)*ParticleColor.a;
//    color.b = color.b + f*2;
    if(Temperature < 3) color.a = 1 - f*10;
    else color.a = 1;
    if(f>(0.1 + 0.5*Temperature/5)*(0.1 + 0.5*Temperature/5)) discard;

//    color = vec4(1,1,1,1);
}
