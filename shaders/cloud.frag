#version 330 core

in vec2 TexCoords;
in float humidity;
in float condensation;
in float x_off;
in float y_off;
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

// p must be normalized!
float perlin(vec2 p, float dim) {
        return perlin(p, dim, 0.0);
}


void main(){
    vec2 temp = TexCoords - vec2(0.5);
    float f = dot(temp, temp);
    float w = perlin(TexCoords+vec2(x_off, y_off), 2)/2;
    color = vec4(0.5+w,0.5+w,0.5+w,(w+0.5)*(1-f*4));
    if(f>0.25) discard;
}
