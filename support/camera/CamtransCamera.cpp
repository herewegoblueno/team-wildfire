/**
 * @file   CamtransCamera.cpp
 *
 * This is the perspective camera class you will need to fill in for the Camtrans lab.  See the
 * lab handout for more details.
 */

#include "CamtransCamera.h"
#include "support/Settings.h"

CamtransCamera::CamtransCamera()
{
    m_near = 1;
    m_far = 30;
    setHeightAngle(60);
    setAspectRatio(1);
    m_eye = glm::vec4(2,2,2,1);
    orientLook(getPosition(), -getPosition(), glm::vec4(0, 1, 0, 0));
}

void CamtransCamera::setAspectRatio(float a)
{
    m_aspectRatio = a;
    float heightOverTwo = m_far * tan(glm::radians(m_thetaH/2.f));
    float widthOverTwo = a * heightOverTwo;
    m_thetaW = glm::degrees(2 * atan(widthOverTwo / m_far));
    updateProjectionMatrix();
}

glm::vec4 CamtransCamera::getPosition() const {
    return m_eye;
}

glm::vec4 CamtransCamera::getLook() const {
    return -m_w;
}

glm::vec4 CamtransCamera::getUp() const {
    return m_up;
}

glm::vec4 CamtransCamera::getU() {
    return m_u;
}


glm::vec4 CamtransCamera::getW() {
    return m_w;
}


glm::vec4 CamtransCamera::getV() {
    return m_v;
}


float CamtransCamera::getAspectRatio() const {
    return m_aspectRatio;
}

float CamtransCamera::getHeightAngle() const {
   return m_thetaH;
}

void CamtransCamera::orientLook(const glm::vec4 &eye, const glm::vec4 &look, const glm::vec4 &up) {
   m_eye = eye;
   m_w = glm::normalize(-look);
   m_up = up;
   m_v = glm::normalize(m_up - glm::dot(glm::vec3(m_up), glm::vec3(m_w)) * m_w);
   m_u = glm::normalize(glm::vec4(glm::cross(glm::vec3(m_v), glm::vec3(m_w)), 0));
   updateProjectionMatrix();
   updateViewMatrix();
}

void CamtransCamera::setHeightAngle(float h) {
    m_thetaH = h;
    setAspectRatio(m_aspectRatio); //will recalculate theta of w

}

void CamtransCamera::translate(const glm::vec4 &v) {
    m_eye = v + m_eye;
    updateViewMatrix();
}

void CamtransCamera::rotateU(float degrees) {
    float rad = glm::radians(degrees);
    glm::vec4 newV = m_w * glm::sin(rad) + m_v * glm::cos(rad);
    glm::vec4 newW = m_w * glm::cos(rad) - m_v * glm::sin(rad);
    m_v = newV;
    m_w = newW;
    updateViewMatrix();
}

void CamtransCamera::rotateV(float degrees) {
    float rad = glm::radians(degrees);
    glm::vec4 newU = m_u * glm::cos(rad) - m_w * glm::sin(rad);
    glm::vec4 newW = m_u * glm::sin(rad) + m_w * glm::cos(rad);
    m_u = newU;
    m_w = newW;
    updateViewMatrix();
}

void CamtransCamera::rotateW(float degrees) {
    float rad = -glm::radians(degrees);
    glm::vec4 newU = m_u * glm::cos(rad) - m_v * glm::sin(rad);
    glm::vec4 newV = m_u * glm::sin(rad) + m_v * glm::cos(rad);
    m_u = newU;
    m_v = newV;
    updateViewMatrix();
}

void CamtransCamera::setClip(float nearPlane, float farPlane) {
    m_near = nearPlane;
    m_far = farPlane;
    updateProjectionMatrix();
}


void CamtransCamera::updateProjectionMatrix(){
    updatePerspectiveMatrix();
    updateScaleMatrix();
}

glm::mat4x4 CamtransCamera::getProjectionMatrix() const {
    return  getPerspectiveMatrix() * m_scaleMatrix;
}

void CamtransCamera::updateViewMatrix(){
    updateRotationMatrix();
    updateTranslationMatrix();
}

glm::mat4x4 CamtransCamera::getViewMatrix() const {
    return m_rotationMatrix * m_translationMatrix;
}

void CamtransCamera::updatePerspectiveMatrix(){
    float c = -m_near/m_far;
    m_perspectiveTransformation = glm::mat4x4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, -1/(1+c), -1,
                0, 0, c/(1+c), 0);;
}

glm::mat4x4 CamtransCamera::getPerspectiveMatrix() const {
    return m_perspectiveTransformation;
}

void CamtransCamera::updateScaleMatrix(){
    m_scaleMatrix = glm::mat4x4(
                1/(float)(m_far * tan(glm::radians(m_thetaW)/2)), 0, 0, 0,
                0, 1/(float)(m_far * tan(glm::radians(m_thetaH)/2)), 0, 0,
                0, 0, 1/(float)m_far, 0,
                0, 0, 0, 1
                );
}

glm::mat4x4 CamtransCamera::getScaleMatrix() const {
    return m_scaleMatrix;
}


void CamtransCamera::updateRotationMatrix(){
    m_rotationMatrix = glm::mat4x4(
                m_u.x, m_v.x, m_w.x, 0,
                m_u.y, m_v.y, m_w.y, 0,
                m_u.z, m_v.z, m_w.z, 0,
                0, 0, 0, 1
                );
}

void CamtransCamera::updateTranslationMatrix(){
    m_translationMatrix = glm::mat4x4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -getPosition().x, -getPosition().y, -getPosition().z, 1
                );
}

