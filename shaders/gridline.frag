#version 330 core
in vec3 worldPosition;

out vec4 FragColor;
uniform vec4 color;
uniform vec3 camPos;
void main()
{
   float optimalDistance = 4;
   float viewWindowWhenCloser = 0.2;
   float viewWindowWhenFurther = 0.2;

   float distance = length(camPos - worldPosition);
   float differenceFromOptimal = abs(distance - optimalDistance);
   float alpha = color.w;

   if (distance > optimalDistance){
       alpha *= max(0, 1 - differenceFromOptimal / viewWindowWhenFurther);
   }else{
       alpha *= max(0, 1 - differenceFromOptimal / viewWindowWhenCloser);
   }
   FragColor = vec4(color.xyz, alpha);
}
