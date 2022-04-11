#ifndef ORBITINGCAMERA_H
#define ORBITINGCAMERA_H

#include "Camera.h"


/**
 * @class OrbitingCamera
 *
 * Camera that moves in response to mouse interaction.
 *
 * You shouldn't need to work on this class. It's there for your convenience, really,
 * and the way this camera is implemented is NOT the way you should be implementing
 * your camera in the Camtrans lab. We hide the real implementation by using OpenGL to
 * perform the camera calculations.
 *
 */
class OrbitingCamera : public Camera {
public:
    OrbitingCamera() : m_angleX(0), m_angleY(0), m_zoomZ(-5) {m_aspectRatio = 1;}
    ~OrbitingCamera() {}

    virtual void setAspectRatio(float aspectRatio);

    virtual glm::mat4x4 getProjectionMatrix() const;
    virtual glm::mat4x4 getViewMatrix() const;
    virtual glm::mat4x4 getScaleMatrix() const;
    virtual glm::mat4x4 getPerspectiveMatrix() const;

    virtual void mouseDown(int x, int y);
    virtual void mouseDragged(int x, int y);
    virtual void mouseScrolled(int delta);

    void updateMatrices();

private:
    void updateProjectionMatrix();
    void updateViewMatrix();

    glm::mat4x4 m_viewMatrix;
    glm::mat4x4 m_projectionMatrix;
    glm::mat4x4 m_scaleMatrix;
    float m_angleX, m_angleY, m_zoomZ;
    int m_oldX, m_oldY;
};

#endif // ORBITINGCAMERA_H
