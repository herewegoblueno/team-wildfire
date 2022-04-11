#include "SupportCanvas3D.h"

#include <QFileDialog>
#include <QMouseEvent>
#include <QMessageBox>
#include <QApplication>

#include "support/Settings.h"
#include "support/lib/RGBA.h"
#include "support/camera/CamtransCamera.h"
#include "support/camera/OrbitingCamera.h"

#include "BasicForestScene.h"

#include <iostream>
#include "support/gl/GLDebug.h"
#include "support/lib/CS123XmlSceneParser.h"

//This has been designed so that this one canvas can render differnet things and use differnt camera settings based on what the
//current "scene" is
SupportCanvas3D::SupportCanvas3D(QGLFormat format, MainWindow *parent) : QGLWidget(format, parent),
    m_isDragging(false),
    m_settingsDirty(true),
    m_defaultPerspectiveCamera(new CamtransCamera()),
    m_defaultOrbitingCamera(new OrbitingCamera()),
    m_currentScene(nullptr),
    m_mainWindowParent(parent)

{
    //The CameraConfig for the FOREST_SCENE_MODE is set up in MainWindow::fileOpen
}

SupportCanvas3D::~SupportCanvas3D()
{
}

Camera *SupportCanvas3D::getCamera() {
    switch(settings.getCameraMode()) {
        case CAMERAMODE_CAMTRANS:
            return m_defaultPerspectiveCamera.get();

        case CAMERAMODE_ORBIT:
            return m_defaultOrbitingCamera.get();

        default:
            return nullptr;
    }
}

OrbitingCamera *SupportCanvas3D::getOrbitingCamera() {
    return m_defaultOrbitingCamera.get();
}


CameraConfig *SupportCanvas3D::getCurrentSceneCamtasConfig() {
    switch(settings.getSceneMode()) {
        case FOREST_SCENE_MODE:
           return &m_basicForestSceneCameraConfig;
    }
    return nullptr;
}

void SupportCanvas3D::initializeGL() {
    // Track the camera settings so we can generate deltas
    m_oldPosX = settings.cameraPosX;
    m_oldPosY = settings.cameraPosY;
    m_oldPosZ = settings.cameraPosZ;
    m_oldRotU = settings.cameraRotU;
    m_oldRotV = settings.cameraRotV;
    m_oldRotN = settings.cameraRotN;

    initializeGlew();

    initializeOpenGLSettings();
    initializeScenes();
    setSceneFromSettings();

    settingsChanged();

    m_mainWindowParent->onSupportCanvasInitialized();
}

void SupportCanvas3D::initializeGlew() {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    glGetError(); // Clear errors after call to glewInit
    if (GLEW_OK != err) {
      // Problem: glewInit failed, something is seriously wrong.
      fprintf(stderr, "Error initializing glew: %s\n", glewGetErrorString(err));
    }
}

void SupportCanvas3D::initializeOpenGLSettings() {
    // Enable depth testing, so that objects are occluded based on depth instead of drawing order.
    glEnable(GL_DEPTH_TEST);

    // Move the polygons back a bit so lines are still drawn even though they are coplanar with the
    // polygons they came from, which will be drawn before them.
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1, -1);

    // Enable back-face culling, meaning only the front side of every face is rendered.
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    // Specify that the front face is represented by vertices in counterclockwise order (this is
    // the default).
    glFrontFace(GL_CCW);

    // Calculate the orbiting camera matrices.
    getOrbitingCamera()->updateMatrices();
}

void SupportCanvas3D::initializeScenes() {
    m_basicForestScene = std::make_unique<BasicForestScene>();
}

void SupportCanvas3D::paintGL() {
    if (m_settingsDirty) {
        setSceneFromSettings();
    }

    float ratio = static_cast<QGuiApplication *>(QCoreApplication::instance())->devicePixelRatio();
    glViewport(0, 0, width() * ratio, height() * ratio);
    getCamera()->setAspectRatio(static_cast<float>(width()) / static_cast<float>(height()));
    m_currentScene->render(this);
}

void SupportCanvas3D::settingsChanged() {
    m_settingsDirty = true;
    if (m_currentScene != nullptr) {
        // Just calling this function so that the scene is always updated.
        setSceneFromSettings();
        m_currentScene->settingsChanged();
    }
    update(); /* repaint the scene */
}

void SupportCanvas3D::setSceneFromSettings() {
    switch(settings.getSceneMode()) {
        case FOREST_SCENE_MODE:
            setSceneToForestScene();
            break;
    }
    m_settingsDirty = false;
}

void SupportCanvas3D::loadSceneFromParserForForestScene(CS123XmlSceneParser &parser) {
    assert(settings.getSceneMode() == FOREST_SCENE_MODE);
    //The basic forest scene is currently the only scene who is designed to load from an xml file
    m_basicForestScene = std::make_unique<BasicForestScene>();
    Scene::parse(m_basicForestScene.get(), &parser);
    m_basicForestScene.get()->onNewSceneLoaded();
    applyCameraConfig(m_basicForestSceneCameraConfig);
    m_settingsDirty = true;
}

void SupportCanvas3D::applyCameraConfig(CameraConfig c){
    m_defaultPerspectiveCamera->orientLook(c.pos, c.look, c.up);
    m_defaultPerspectiveCamera->setHeightAngle(c.angle);
}



void SupportCanvas3D::setSceneToForestScene(){
    assert(m_basicForestScene.get());
    m_currentScene = m_basicForestScene.get();
    applyCameraConfig(m_basicForestSceneCameraConfig);
}


void SupportCanvas3D::copyPixels(int width, int height, RGBA *data) {
    glReadPixels(0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, data);
    std::cout << "copied " << width << "x" << height << std::endl;

    // Flip the image and since OpenGL uses an origin in the lower left and we an origin in the
    // upper left.
    for (int y = 0; y < (height + 1) / 2; y++)
        for (int x = 0; x < width; x++)
            std::swap(data[x + y * width], data[x + (height - y - 1) * width]);
}

void SupportCanvas3D::resetUpVector() {
    // Reset the up vector to the y axis
    glm::vec4 up = glm::vec4(0.f, 1.f, 0.f, 0.f);
    if (fabs(glm::length(m_defaultPerspectiveCamera->getUp() - up)) > 0.0001f) {
        m_defaultPerspectiveCamera->orientLook(
                    m_defaultPerspectiveCamera->getPosition(),
                    m_defaultPerspectiveCamera->getLook(),
                    up);
        update();
    }
}


void SupportCanvas3D::setCameraAxisX() {
    m_defaultPerspectiveCamera->orientLook(
                glm::vec4(2.f, 0.f, 0.f, 1.f),
                glm::vec4(-1.f, 0.f, 0.f, 0.f),
                glm::vec4(0.f, 1.f, 0.f, 0.f));
    update();
}

void SupportCanvas3D::setCameraAxisY() {
    m_defaultPerspectiveCamera->orientLook(
                glm::vec4(0.f, 2.f, 0.f, 1.f),
                glm::vec4(0.f, -1.f, 0.f, 0.f),
                glm::vec4(0.f, 0.f, 1.f, 0.f));
    update();
}

void SupportCanvas3D::setCameraAxisZ() {
    m_defaultPerspectiveCamera->orientLook(
                glm::vec4(0.f, 0.f, 2.f, 1.f),
                glm::vec4(0.f, 0.f, -1.f, 0.f),
                glm::vec4(0.f, 1.f, 0.f, 0.f));
    update();
}

void SupportCanvas3D::setCameraAxonometric() {
    m_defaultPerspectiveCamera->orientLook(
                glm::vec4(2.f, 2.f, 2.f, 1.f),
                glm::vec4(-1.f, -1.f, -1.f, 0.f),
                glm::vec4(0.f, 1.f, 0.f, 0.f));
    update();
}

void SupportCanvas3D::updateCameraHeightAngle() {
    // The height angle is half the overall field of view of the camera
    m_defaultPerspectiveCamera->setHeightAngle(settings.cameraFov);
}

void SupportCanvas3D::updateCameraTranslation() {
    m_defaultPerspectiveCamera->translate(
            glm::vec4(
                settings.cameraPosX - m_oldPosX,
                settings.cameraPosY - m_oldPosY,
                settings.cameraPosZ - m_oldPosZ,
                0));

    m_oldPosX = settings.cameraPosX;
    m_oldPosY = settings.cameraPosY;
    m_oldPosZ = settings.cameraPosZ;
}

void SupportCanvas3D::updateCameraRotationU() {
    m_defaultPerspectiveCamera->rotateU(settings.cameraRotU - m_oldRotU);
    m_oldRotU = settings.cameraRotU;
}

void SupportCanvas3D::updateCameraRotationV() {
    m_defaultPerspectiveCamera->rotateV(settings.cameraRotV - m_oldRotV);
    m_oldRotV = settings.cameraRotV;
}

void SupportCanvas3D::updateCameraRotationN() {
    m_defaultPerspectiveCamera->rotateW(settings.cameraRotN - m_oldRotN);
    m_oldRotN = settings.cameraRotN;
}

void SupportCanvas3D::updateCameraClip() {
    m_defaultPerspectiveCamera->setClip(settings.cameraNear, settings.cameraFar);
}


void SupportCanvas3D::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::RightButton) {
        getCamera()->mouseDown(event->x(), event->y());
        m_isDragging = true;
        update();
    }
}

void SupportCanvas3D::mouseMoveEvent(QMouseEvent *event) {
    if (m_isDragging) {
        getCamera()->mouseDragged(event->x(), event->y());
        update();
    }
}

void SupportCanvas3D::mouseReleaseEvent(QMouseEvent *event) {
    if (m_isDragging && event->button() == Qt::RightButton) {
        getCamera()->mouseUp(event->x(), event->y());
        m_isDragging = false;
        update();
    }
}

void SupportCanvas3D::wheelEvent(QWheelEvent *event) {
    getCamera()->mouseScrolled(event->delta());
    update();
}

void SupportCanvas3D::resizeEvent(QResizeEvent *event) {
    emit aspectRatioChanged();
}
