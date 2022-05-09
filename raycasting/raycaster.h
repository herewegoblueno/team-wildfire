#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "GL/glew.h"
#include "support/camera/Camera.h"

struct Ray{
    glm::vec3 source;
    glm::vec3 direction;
};

struct Plane{
    glm::vec3 samplePoint;
    glm::vec3 normal;
};

struct IntersectionStatus{
    bool intersectFound;
    glm::vec3 point;
};



class Raycaster
{
public:
    static IntersectionStatus checkMouseClickIntersectionWithPlane(Camera *cam, float mouseXFilm, float mouseYFilm, Plane plane);
private:
    static Ray generateWorldSpaceRayFromMouseClick(Camera *camera, float mouseXFilm, float mouseYFilm);
    static float checkPlaneIntersection(Ray ray, Plane plane);
    static glm::vec3 genPoint(Ray ray, float t);
};

#endif // RAYCASTER_H
