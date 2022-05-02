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

    if(Temperature>4.2) color = vec4(0.7, 0.4, 0.3, 1);
    else if(Temperature>3.9) color = vec4(0.8, 0.4, 0.3, 1);
    else if(Temperature>3.6) color = vec4(0.8, 0.25, 0.2, 1);
    else if(Temperature>3.4) color = vec4(0.9, 0.1, 0.0, 1);
    else color = vec4(0.8, 0, 0, 1);

    //For some reason this isn't working on Mac....
    color = (texture(sprite, TexCoords)+0.2)*color;

    if(f>(0.1 + 0.5*Temperature/5)*(0.1 + 0.5*Temperature/5)) discard;
}
