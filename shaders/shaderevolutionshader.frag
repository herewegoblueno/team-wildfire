#version 330 core

in vec3 color;
in vec3 pos; //(object space)
in float timevar;

out vec4 fragColor;

//Not in use at the moment
vec3 rangeLimit(vec3 v){
    v.x = 1.f / (1 + exp(- 0.6 * v.x));
    v.y = 1.f / (1 + exp(- 0.6 * v.y));
    v.z = 1.f / (1 + exp(- 0.6 * v.z));
    return v;
}

//For CrossProductNode
vec3 my_cross(vec3 a, vec3 b){
    vec3 product = cross(a, b);
    product = normalize(product);
    product *= (length(a) + length(b)) / 2.f;
    return product;
}

//Credit for the next 3 functions (for perlin noise):
//https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83

float rand(vec2 c){
        return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float noise(vec2 p, float freq ){
        float unit = 2/freq;
        vec2 ij = floor(p/unit);
        vec2 xy = mod(p,unit)/unit;
        xy = 0.5 * (1.0 -cos(3.1415 * xy));
        float a = rand((ij+vec2(0.,0.)));
        float b = rand((ij+vec2(1.,0.)));
        float c = rand((ij+vec2(0.,1.)));
        float d = rand((ij+vec2(1.,1.)));
        float x1 = mix(a, b, xy.x);
        float x2 = mix(c, d, xy.x);
        return mix(x1, x2, xy.y);
}

float pNoise(vec2 p, int res){
        float persistance = .5;
        float n = 0.;
        float normK = 0.;
        float f = 4.;
        float amp = 1.;
        int iCount = 0;
        for (int i = 0; i< 50; i++){
                n+=amp*noise(p, f);
                f*=2.;
                normK+=amp;
                amp*=persistance;
                if (iCount == res) break;
                iCount++;
        }
        float nf = n/normK;
        return nf * nf * nf * nf;
}

vec3 perlinNoiseVec3(vec3 v, vec3 f){
    int res = int(length(f) * 5);
    return vec3(pNoise(v.xy, res), pNoise(v.yz, res), pNoise(v.zx, res));
}

//For the 3 transplant nodes...
vec3 transplantY(vec3 donor, vec3 recepeint){
    return vec3(recepeint.x, donor.y, recepeint.z);
}

vec3 transplantX(vec3 donor, vec3 recepeint){
    return vec3(donor.x, recepeint.y, recepeint.z);
}

vec3 transplantZ(vec3 donor, vec3 recepeint){
    return vec3(recepeint.x, recepeint.y, donor.z);
}

//For AverageNode
vec3 average(vec3 a, vec3 b){
    return (a + b) / 2.0;
}

//For JuliaFractalNode
//See README for info on inspiratoin for this methodology
//l[x]v[n] (stand for color layer x vector n) parameters are [0, 4]
//l[x]ci[n] (stand for color layer coorindate index n) parameters are [0, 1]
vec3 fractal(vec3 scalevec, vec3 centervec, vec3 seedx, vec3 seedy,
             bool usePositionForSeed, bool breakAfter, vec3 zStarter,
             int l1v1, int l1v2, int l1v3, int l1ci1, int l1ci2, int l1ci3,
             int l2v1, int l2v2, int l2v3, int l2ci1, int l2ci2, int l2ci3,
             int l3v1, int l3v2, int l3v3, int l3ci1, int l3ci2, int l3ci3){

    int maxIterations = 10;

    float scale = length(scalevec);
    vec2 center = centervec.xy;
    vec2 z, c;

    c.x = length(seedx);
    c.y = length(seedy);

    if (usePositionForSeed){
        c.x += pos.x * scale - center.x;
        c.y += pos.y * scale - center.y;
    }

    z.x = zStarter.x * scale - center.x;
    z.y = zStarter.y * scale - center.y;

    int i;
    vec2 accumulator = vec2(0,0);
    vec2 multiplier = vec2(1,1);
    vec2 differenceHolder = vec2(0,0);

    for(i=0; i<maxIterations; i++) {

        float x = (z.x * z.x - z.y * z.y) + c.x;
        float y = (z.y * z.x + z.x * z.y) + c.y;

        if (!breakAfter) if((x * x + y * y) > 4.0) break;
        z.x = x;
        z.y = y;

        accumulator += z;
        if (x != 0) multiplier.x *= -x;
        if (y != 0) multiplier.y *= y;
        ((i % 2) == 0) ? differenceHolder += z : differenceHolder -= z;

        if((x * x + y * y) > 4.0) break;
    }

    int numberOfLayers = 3;
    float iterationFraction = 1 - i / float(maxIterations);
    vec2 iterationFractionVec = vec2(iterationFraction, iterationFraction);
    vec2 layerOps[5] = vec2[5](accumulator, multiplier, differenceHolder, iterationFractionVec, z);

    vec4 col;
    col = vec4(layerOps[l1v1][l1ci1], layerOps[l1v2][l1ci2], layerOps[l1v3][l1ci3], 1);
    col += vec4(layerOps[l2v1][l2ci1], layerOps[l2v2][l2ci2], layerOps[l2v3][l2ci3], 1);
    col += vec4(layerOps[l3v1][l3ci1], layerOps[l3v2][l3ci2], layerOps[l3v3][l3ci3], 1);

    if (i == maxIterations){
        col += vec4(z.x, z.y, iterationFraction, 1);
        numberOfLayers++;
    }

    //breakAfter tends to make things very bright
    col /= (numberOfLayers + (breakAfter ? 0.7 : 0));
    return vec3(col);
}

void main(){
    //Default value of fragColor
    fragColor = vec4(pos.r - timevar, pos.g + timevar, -timevar + pos.b, 1) / 0.5f;
    fragColor += vec4(color, 1) / 0.5f;
}
