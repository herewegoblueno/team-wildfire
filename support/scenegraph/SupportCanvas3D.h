#ifndef SUPPORTCANVAS3D_H
#define SUPPORTCANVAS3D_H

#include <memory>

#include "GL/glew.h"
#include <QGLWidget>

#include "glm/glm.hpp"

class RGBA;
class Camera;
class OpenGLScene;
class OrbitingCamera;
class CamtransCamera;
class CS123XmlSceneParser;
class ShaderEvolutionTestingScene;
class ShaderImportScene;
class LSystemTreeScene;
class GalleryScene;

/**
 * @class  SupportCanvas3D
 *
 * The SupportCanvas3D class holds a single active OpenGLScene, and either
 * calls upon that scene to draw itself using OpenGL or draws the scene
 * by directly calling upon OpenGL (getting the scene-specific information
 * from the OpenGLScene object). The details of the implementation are left
 * to the student; neither way is better than the other.
 *
 * The SupportCanvas3D also contains a default camera which can be used in
 * case the loaded scene does not specify a camera.
 */

struct CameraConfig {
   glm::vec4 pos;
   glm::vec4 look;
   glm::vec4 up;
  float angle;
};

class SupportCanvas3D : public QGLWidget {
    Q_OBJECT
public:
    SupportCanvas3D(QGLFormat format, QWidget *parent);

    virtual ~SupportCanvas3D();

    Camera *getCamera();
    OrbitingCamera *getOrbitingCamera();
    CameraConfig *getCurrentSceneCamtasConfig();

    // Returns a pointer to the current scene. If no scene is loaded, this function returns nullptr.
    OpenGLScene *getScene() { return m_currentScene; }
    ShaderEvolutionTestingScene *getShaderScene() { return m_shaderTestingScene.get(); }
    ShaderImportScene *getImportScene() { return m_shaderImportScene.get(); }


    void loadSceneFromParser(CS123XmlSceneParser &parser);
    void switchToSceneviewScene();
    void switchToShapesScene();

    // Copies pixels from the OpenGL render buffer into a standard bitmap image, using row-major
    // order and RGBA data format.
    void copyPixels(int width, int height, RGBA *data);

    // This function will be called by the UI when the settings have changed (see mainwindow.cpp)
    virtual void settingsChanged();

public slots:
    // These will be called by the corresponding UI buttons on the Camtrans dock
    // ^from support code, probably won't do this anymore
    void resetUpVector();
    void setCameraAxisX();
    void setCameraAxisY();
    void setCameraAxisZ();
    void setCameraAxonometric();

    // These will be called whenever the corresponding UI elements are updated on the Camtrans dock
    // ^from support code, probably won't do this anymore
    void updateCameraHeightAngle();
    void updateCameraTranslation();
    void updateCameraRotationU();
    void updateCameraRotationV();
    void updateCameraRotationN();
    void updateCameraClip();

signals:
    void aspectRatioChanged();

protected:
    // Overridden from QGLWidget
    virtual void initializeGL() override;
    virtual void paintGL() override;

    // Overridden from QWidget
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;
    virtual void resizeEvent(QResizeEvent *event) override;

    float m_oldPosX, m_oldPosY, m_oldPosZ;
    float m_oldRotU, m_oldRotV, m_oldRotN;

private:

    void initializeGlew();
    void initializeOpenGLSettings();
    void initializeScenes();
    void setSceneFromSettings();
    void setSceneToLSystemSceneview();
    void setSceneToShaderTesting();
    void setSceneToShaderImport();
    void setSceneToGallery();

    void applyCameraConfig(CameraConfig c);


    glm::vec4      m_cameraEye;
    bool           m_isDragging;

    bool m_settingsDirty;

    std::unique_ptr<CamtransCamera> m_defaultPerspectiveCamera;
    std::unique_ptr<OrbitingCamera> m_defaultOrbitingCamera;
    OpenGLScene *m_currentScene;

    std::unique_ptr<ShaderEvolutionTestingScene> m_shaderTestingScene;
    CameraConfig m_shaderTestingSceneCameraConfig;

    std::unique_ptr<ShaderImportScene> m_shaderImportScene;
    CameraConfig m_shaderImportSceneCameraConfig;

    std::unique_ptr<LSystemTreeScene> m_LSystemScene;
    CameraConfig m_LSystemSceneCameraConfig;

    std::unique_ptr<GalleryScene> m_galleryScene;
    CameraConfig m_GallerySceneCameraConfig;

};



#endif // SUPPORTCANVAS3D_H
