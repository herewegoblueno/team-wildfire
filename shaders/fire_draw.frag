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

    if(ParticleLife>4) color = vec4(1, 1, 0.5, 0);
    else if(ParticleLife>3) color = vec4(1, 0.5, 0.3, 0);
    else if(ParticleLife>2) color = vec4(1, 0.2, 0.2, 0);
    else if(ParticleLife>1) color = vec4(0.7, 0.2, 0.1, 0);
    else color = vec4(0.4, 0, 0, 0);

//    color = texture(sprite, TexCoords)*ParticleColor;
//    color = ParticleColor;
//    color.a = (color.x+color.y+color.z)*ParticleColor.a;
    color.a = 1/(f+1)+0.3;
//    if(f>0.25) color.a = 0;
//    color = ParticleColor;
}
