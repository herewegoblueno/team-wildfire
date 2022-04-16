#version 330 core
in vec2 TexCoords;
in vec4 ParticleColor;
out vec4 color;

uniform sampler2D sprite;

void main()
{
    color = texture(sprite, TexCoords)*ParticleColor;
    color.a = (color.x+color.y+color.z)*ParticleColor.a;
//    color.a = 0.3;
//    color = ParticleColor;
}
