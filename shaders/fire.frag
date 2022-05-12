#version 330 core
in vec2 TexCoords;
in float Life;
in float pattern;
in float stamp;

out vec4 color;


float rand(vec2 co){return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);}
float rand (vec2 co, float l) {return rand(vec2(rand(co), l));}
float rand (vec2 co, float l, float t) {return rand(vec2(rand(co, l), t));}

float perlin(vec2 p, float dim, float time) {
        vec2 pos = floor(p * dim);
        vec2 posx = pos + vec2(1.0, 0.0);
        vec2 posy = pos + vec2(0.0, 1.0);
        vec2 posxy = pos + vec2(1.0);

        float c = rand(pos, dim, time);
        float cx = rand(posx, dim, time);
        float cy = rand(posy, dim, time);
        float cxy = rand(posxy, dim, time);

        vec2 d = fract(p * dim);
        d = -0.5 * cos(d * 3.1415926535897932384) + 0.5;

        float ccx = mix(c, cx, d.x);
        float cycxy = mix(cy, cxy, d.x);
        float center = mix(ccx, cycxy, d.y);

        return center * 2.0 - 1.0;
}

void main()
{
    vec2 temp = TexCoords - vec2(0.5);
    float f = dot(temp, temp);
//    if(f>0.25) discard;

    if(Life>4.5) color = vec4(0.7, 0.35-f, 0.1, 1);
    else if(Life>4.0) color = vec4(0.8, 0.28-f, 0.07, 1);
    else if(Life>3.6) color = vec4(0.8, 0.25-f, 0.02, 1);
    else if(Life>3.0) color = vec4(0.8, 0.00, 0.0, 1);
    else color = vec4(0.8, 0, 0, 1);

    //For some reason this isn't working on Mac....
    float guage = (0.5*Life/5)*(0.5*Life/5);
    float w = perlin(TexCoords, 2.5, 5-Life);
    color.a = (w*0.8 + 0.2)*(1-f/(guage+0.04));
    if(f>guage+0.04) discard;
//    color = vec4(1,1,1,1);

}
