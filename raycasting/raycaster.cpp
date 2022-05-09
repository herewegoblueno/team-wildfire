#include "raycaster.h"

IntersectionStatus Raycaster::checkMouseClickIntersectionWithPlane(Camera *cam, float mouseXFilm, float mouseYFilm, Plane plane){
    Ray worldspaceaRay = generateWorldSpaceRayFromMouseClick(cam, mouseXFilm, mouseYFilm);
    float intersection = checkPlaneIntersection(worldspaceaRay, plane);

    if (intersection == std::numeric_limits<float>::infinity()){
        return {false, glm::vec3(0,0,0)};
    }

    return {true, genPoint(worldspaceaRay, intersection)};
}



Ray Raycaster::generateWorldSpaceRayFromMouseClick(Camera *camera, float mouseXFilm, float mouseYFilm){
    //I need the inverse of of the camera's trans * rot * scaling matrix
    glm::mat4x4 m_worldToFilm = camera->getScaleMatrix() * camera->getViewMatrix();
    glm::mat4x4 m_filmToWorld = glm::inverse(m_worldToFilm);

    //Make a ray from origin to film point (where z = -1, x and y = [-1, 1]
    glm::vec3 rayOrigin = glm::vec3(0,0,0);
    glm::vec3 filmPoint = glm::vec3(mouseXFilm, mouseYFilm, -1);

    //transform that to world space by undoing the scaling, rotation and translation
    filmPoint = glm::vec3(m_filmToWorld * glm::vec4(filmPoint, 1.f));
    rayOrigin = glm::vec3(m_filmToWorld * glm::vec4(rayOrigin, 1.f));
    return {rayOrigin, glm::normalize(filmPoint - rayOrigin)};
}

//Returns infinity if there was no intersection or if the intersection is behind the user
//Otherwise, returns t, which is the distance down the ray to the point
float Raycaster::checkPlaneIntersection(Ray ray, Plane plane){
    float dotDemoninator = glm::dot(ray.direction, plane.normal);
    if (dotDemoninator != 0){
        float tempT = glm::dot((plane.samplePoint - ray.source), plane.normal) / dotDemoninator;
        if (tempT > 0) return tempT;
    }
    return std::numeric_limits<float>::infinity();
};


glm::vec3 Raycaster::genPoint(Ray ray, float t){
    return ray.source + ray.direction * t;
}


