/**
 * @file   OrbitingCamera.cpp
 *
 * (See the header file.) You don't need to be poking around in this file unless you're interested
 * in how an orbiting camera works.
 *
 * The way we have implemented this class is NOT how you should be implementing your Camtrans. This
 * camera is a DIFFERENT TYPE of camera which we're providing so you can easily view your Shapes
 * and to make sure your scene graph is working if your camera isn't.
 *
 * In the Camtrans lab, you'll be implementing your own perspective camera from scratch! This one
 * uses OpenGL.
 */

#include <float.h>
#include <math.h>

#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"

#include "OrbitingCamera.h"
#include "support/Settings.h"


void OrbitingCamera::setAspectRatio(float aspectRatio) {
    m_aspectRatio = aspectRatio;
    updateProjectionMatrix();
}

glm::mat4x4 OrbitingCamera::getProjectionMatrix() const {
    return m_projectionMatrix;
}

glm::mat4x4 OrbitingCamera::getViewMatrix() const {
    return m_viewMatrix;
}

glm::mat4x4 OrbitingCamera::getScaleMatrix() const {
    return m_scaleMatrix;
}

glm::mat4x4 OrbitingCamera::getPerspectiveMatrix() const {
    throw 0; // not implemented
}

void OrbitingCamera::mouseDown(int x, int y) {
    m_oldX = x;
    m_oldY = y;
}

void OrbitingCamera::mouseDragged(int x, int y) {
    m_angleY += x - m_oldX;
    m_angleX += y - m_oldY;
    m_oldX = x;
    m_oldY = y;
    if (m_angleX < -90) m_angleX = -90;
    if (m_angleX > 90) m_angleX = 90;

    updateViewMatrix();
}

void OrbitingCamera::mouseScrolled(int delta) {
    // Use an exponential factor so the zoom increments are small when we are
    // close to the object and large when we are far away from the object
    m_zoomZ *= powf(0.999f, delta);

    updateViewMatrix();
}

void OrbitingCamera::updateMatrices() {
    updateProjectionMatrix();
    updateViewMatrix();
}

void OrbitingCamera::updateProjectionMatrix() {
    // Make sure glm gets a far value that is greater than the near value.
    // Thanks Windows for making far a keyword!
    float farPlane = std::max(settings.cameraFar, settings.cameraNear + 100.f * FLT_EPSILON);
    float h = farPlane * glm::tan(glm::radians(settings.cameraFov * 0.5f));
    float w = m_aspectRatio * h;

    m_scaleMatrix = glm::scale(glm::vec3(1.f / w, 1.f / h, 1.f / farPlane));
    m_projectionMatrix = glm::perspective(
            glm::radians(settings.cameraFov), m_aspectRatio, settings.cameraNear, farPlane) * 0.02f;
}

void OrbitingCamera::updateViewMatrix() {
    m_viewMatrix =
            glm::translate(glm::vec3(0.f, 0.f, m_zoomZ)) *
            glm::rotate(glm::radians(m_angleY), glm::vec3(0.f, 1.f, 0.f)) *
            glm::rotate(glm::radians(m_angleX), glm::vec3(1.f, 0.f, 0.f));
}
