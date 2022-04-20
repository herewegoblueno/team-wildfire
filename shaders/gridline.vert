#version 330 core

 layout (location = 0) in vec3 aPos;
 out vec3 worldPosition;

 uniform mat4 PV;
 uniform vec3 m;

 uniform bool renderingVectorField;
 uniform vec3 u; //The value of the sample of the vector field we're rendering (if we're rendering one)

 void main()
 {
    vec3 inputPosition = aPos;

    if (renderingVectorField){
        if (length(u) == 0){
            //Hide the line if there's no u
            inputPosition *= length(u);
        }else if (normalize(u) == vec3(0,-1,0)){
            //The field is parallel to the input vector we use to render the field, we just have to flip you
            inputPosition *= -1;
        }else if (normalize(u) != vec3(0,1,0)){
            //So far, the field is non-zero, is not parallel to the vector we use to render the field,
            //and also isn't equal to said vector (which is 0, 1, 0), so...
            //Make a rotation matrix (R) to align the input line (0,1,0) to the acutal u sample
            //https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            vec3 a = vec3(0,1,0);
            vec3 b = normalize(u);
            vec3 v = cross(a, b);
            float c = dot(a, b);
            mat3 identity = mat3(1.0);
            mat3 xv;
            xv[0] = vec3(0, v.z, -v.y); // this sets a column, not a row
            xv[1] = vec3(-v.z, 0, v.x);
            xv[2] = vec3(v.y, -v.x, 0);

            mat3x3 R = identity + xv + (xv * xv) * (1 / (1 + c));
            inputPosition = R * inputPosition;
        }
    }

    worldPosition = inputPosition + m;
    gl_Position =  PV * vec4(worldPosition, 1.0);
 }
