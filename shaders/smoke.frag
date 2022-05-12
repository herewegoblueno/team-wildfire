#version 330 core
in vec2 TexCoords;
in float Temperature;
out vec4 color;

uniform float base_x;
uniform float base_y;


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
    if(f>0.25) discard;

    float ref_temp = 20. - Temperature;
    if(ref_temp>4) color = vec4(0.7, 0.7, 0.7, 1);
    else if(ref_temp>3) color = vec4(0.6, 0.6, 0.6, 1);
    else if(ref_temp>2) color = vec4(0.5, 0.5, 0.5, 1);
    else if(ref_temp>1) color = vec4(0.4, 0.4, 0.4, 1);

    float w = perlin(TexCoords+vec2(base_x, base_y)*5, 2.5, 0.);
    color.a = w*(0.25-f);

//    if(f>0.1) discard;
    if(f>0.03*ref_temp) discard;
//    if(f>0.25) discard;

}
