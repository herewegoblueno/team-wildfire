#version 330 core
in vec2 TexCoords;
in vec4 ParticleColor;
in float ParticleLife;
out vec4 color;

uniform sampler2D sprite;

void main()
{
    vec2 temp = TexCoords - vec2(0.5);
    float f = dot(temp, temp);

//    if(ParticleLife>4) color = vec4(0.4, 0.4, 0.4, 0);
//    if(ParticleLife>3) color = vec4(0.3, 0.3, 0.3, 0);
//    if(ParticleLife>2) color = vec4(0.2, 0.2, 0.2, 0);
//    if(ParticleLife>1) color = vec4(0.1, 0.1, 0.1, 0);

//    color = (texture(sprite, TexCoords)+0.2)*color;
//    color = ParticleColor;
//    color.a = (color.x+color.y+color.z)*ParticleColor.a;
//    color.b = color.b + f*2;
//    if(ParticleLife > 3) color.a = 1 - f*6;
    color = vec4(0.3, 0.3, 0.3, 0.5);

    if(f>0.025*ParticleLife) discard;
    if(f>0.25) discard;

//    color = vec4(1,1,1,1);
}
