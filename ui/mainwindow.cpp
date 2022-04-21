#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "GL/glew.h"
#include "support/Settings.h"
#include <qgl.h>
#include <QGLFormat>
#include "support/scenegraph/SupportCanvas3D.h"
#include <QFileDialog>
#include <QMessageBox>
#include "support/camera/CamtransCamera.h"
#include "support/lib/CS123XmlSceneParser.h"
#include <chrono>
#include "voxels/voxelgridline.h"

using namespace std::chrono;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    settings.loadSettingsOrDefaults();

    //You can make changes to the scene now

    //Adding in the 3D canvas...
    QGLFormat qglFormat;
    qglFormat.setVersion(4, 3);
    qglFormat.setProfile(QGLFormat::CoreProfile);
    qglFormat.setSampleBuffers(true);

    m_canvas3D = new SupportCanvas3D(qglFormat, this);
    ui->CanvasParent->addWidget(m_canvas3D, 0, 0);

    //Filling the UI elements with the current values from the settings global variable
    ui->useLightingForShaders->setCheckState((settings.useLighting && settings.usePointLights) ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
    ui->mainTabWidget->setCurrentIndex(settings.currentTab);
    ui->useOrbitingCamera->setCheckState(settings.useOrbitCamera ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
    ui->vizualizeForestVoxelGrid->setCheckState(settings.visualizeForestVoxelGrid ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
    ui->vizualizeForestField->setCheckState(settings.visualizeVectorField ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);

    ui->forestVisualizationEyeXSlider->setRange(-50, 50);
    ui->forestVisualizationEyeXSlider->setValue(settings.visualizeForestVoxelGridEyeX * 10);
    ui->forestVisualizationEyeYSlider->setRange(-50, 50);
    ui->forestVisualizationEyeYSlider->setValue(settings.visualizeForestVoxelGridEyeY * 10);
    ui->forestVisualizationEyeZSlider->setRange(-50, 50);
    ui->forestVisualizationEyeZSlider->setValue(settings.visualizeForestVoxelGridEyeZ * 10);
    ui->forestVisualizationEyeRSlider->setRange(0, 30);
    ui->forestVisualizationEyeRSlider->setValue(settings.visualizeForestVoxelGridEyeRadius * 10);
    ui->visualizationTemperatureRangeSlider->setRange(0, 50);
    ui->visualizationTemperatureRangeSlider->setValues(settings.visualizeForestVoxelGridMinTemp * 10, settings.visualizeForestVoxelGridMaxTemp * 10);

    ui->FieldVisOptionsDropbox->setCurrentIndex(settings.vectorGridMode);
    ui->VoxelVisOptionsDropbox->setCurrentIndex(settings.voxelGridMode);
    ui->seeBranchModules->setChecked(settings.seeBranchModules);

    ui->TimescaleSlider->setRange(0, 20);
    ui->TimescaleSlider->setValue(settings.simulatorTimescale * 10);

    #ifdef QT_DEBUG
      ui->DebugBuildWarning->show();
    #else
      ui->DebugBuildWarning->hide();
    #endif
}

MainWindow::~MainWindow()
{
    delete ui;
}

//Called when the window is closed
void MainWindow::closeEvent(QCloseEvent *event) {
    settings.saveSettings();
    QMainWindow::closeEvent(event);
}

//m_canvas3D doesn't call initializeGL immediately, but we need to wait before it does
//for us to start rendering anything
void MainWindow::onSupportCanvasInitialized(){
    openXmlFileForForestScene(":/xmlScenes/xmlScenes/basicScene.xml");
    initializeFrameCounting();
}


void MainWindow::signalSettingsChanged() {
    //The canvas contains scenes, it'll call settingsChanged in those scenes so don't worry about that
    m_canvas3D->settingsChanged();
}

void MainWindow::notifyFrameCompleted(){
    numberOfFramesRecorded ++;
    int currentTime = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
    int timeDiff = currentTime - timeSinceStartedCounting;
    double averageFrameDuration = timeDiff / (double) numberOfFramesRecorded;
    ui->fpscounter->setText("Avg. FPS: " + QString::number(1 / (double)averageFrameDuration));
}

void MainWindow::initializeFrameCounting(){
    numberOfFramesRecorded = 0;
    timeSinceStartedCounting = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}


void MainWindow::openXmlFileForForestScene(QString file) {
    if (!file.isNull()) {
        if (file.endsWith(".xml")) {
            CS123XmlSceneParser parser(file.toLatin1().data());
            if (parser.parse()) {
                // Set up the camera
                CS123SceneCameraData camera;
                if (parser.getCameraData(camera)) {
                    camera.pos[3] = 1;
                    camera.look[3] = 0;
                    camera.up[3] = 0;

                    CameraConfig *cam = m_canvas3D->getCurrentSceneCamtasConfig();
                    cam->look = camera.look;
                    cam->pos = camera.pos;
                    cam->up = camera.up;
                    cam->angle = camera.heightAngle;
                }
                m_canvas3D->loadSceneFromParserForForestScene(parser);

            } else {
                QMessageBox::critical(this, "Error", "Could not load scene \"" + file + "\"");
            }
        }
        else {
             QMessageBox::critical(this, "Error", "We don't support non-xml stuff yettt");
        }
    }
}

//The following funcitons are autogenerated to link UI elements to the C++ backend
//To make one, go to the designer view of our UI, right click a UI element and click
//"Go to slots"
void MainWindow::on_useLightingForShaders_stateChanged(int state)
{
    settings.useLighting = state == Qt::CheckState::Checked;
    settings.usePointLights = state == Qt::CheckState::Checked;
    signalSettingsChanged();
}

void MainWindow::on_mainTabWidget_currentChanged(int index)
{
    settings.currentTab = index;
    initializeFrameCounting();
    signalSettingsChanged();
}

void MainWindow::changeCameraSettings(bool useOrbiting){
    settings.useOrbitCamera = useOrbiting;
    Qt::CheckState state = useOrbiting ? Qt::CheckState::Checked : Qt::CheckState::Unchecked;
    ui->useOrbitingCamera->setCheckState(state);
    signalSettingsChanged();
}

void MainWindow::on_vizualizeForestVoxelGrid_stateChanged(int state)
{
    settings.visualizeForestVoxelGrid = state == Qt::CheckState::Checked;
    signalSettingsChanged();
}

void MainWindow::on_vizualizeForestField_stateChanged(int state)
{
    settings.visualizeVectorField = state == Qt::CheckState::Checked;
    signalSettingsChanged();
}


void MainWindow::on_useSmoke_stateChanged(int arg1)
{

}


void MainWindow::on_useOrbitingCameraFire_stateChanged(int arg1)
{

}


void MainWindow::on_forestVisualizationEyeXSlider_valueChanged(int value)
{
    ui->forestVisualizationEyeXValue->setText(QString::number(value / 10.0));
    settings.visualizeForestVoxelGridEyeX = value / 10.0;
    signalSettingsChanged();
}


void MainWindow::on_forestVisualizationEyeYSlider_valueChanged(int value)
{
    ui->forestVisualizationEyeYValue->setText(QString::number(value / 10.0));
    settings.visualizeForestVoxelGridEyeY = value / 10.0;
    signalSettingsChanged();
}


void MainWindow::on_forestVisualizationEyeZSlider_valueChanged(int value)
{
    ui->forestVisualizationEyeZValue->setText(QString::number(value / 10.0));
    settings.visualizeForestVoxelGridEyeZ = value / 10.0;
    signalSettingsChanged();
}


void MainWindow::on_forestVisualizationEyeRSlider_valueChanged(int value)
{
    ui->forestVisualizationEyeRValue->setText(QString::number(value / 10.0));
    settings.visualizeForestVoxelGridEyeRadius = value / 10.0;
    signalSettingsChanged();
}


void MainWindow::on_visualizationTemperatureRangeSlider_valuesChanged(int min, int max)
{
    settings.visualizeForestVoxelGridMinTemp = min / 10.0;
    settings.visualizeForestVoxelGridMaxTemp = max / 10.0;
    ui->forestVisualizationTemperatureMinValue->setText(QString::number(min / 10.0));
    ui->forestVisualizationTemperatureMaxValue->setText(QString::number(max / 10.0));
    signalSettingsChanged();
}

void MainWindow::on_VoxelVisOptionsDropbox_currentIndexChanged(int index)
{
    settings.voxelGridMode = static_cast<VoxelVisualizationModes>(index);
    signalSettingsChanged();
}


void MainWindow::on_FieldVisOptionsDropbox_currentIndexChanged(int index)
{
    settings.vectorGridMode = static_cast<VectorFieldVisualizationModes>(index);
    auto explanation = VoxelGridLine::getVectorFieldModeExplanation(static_cast<VectorFieldVisualizationModes>(index));
    ui->VectorFieldExplanation->setText(QString::fromStdString(explanation));
    signalSettingsChanged();
}

void MainWindow::toggleSeeBranchModules() {
    settings.seeBranchModules = !settings.seeBranchModules;
}

void MainWindow::on_TimescaleSlider_valueChanged(int value)
{
    settings.simulatorTimescale = value / 10.f;
    ui->timescaleValue->setText(QString::number(value / 10.f));
}


void MainWindow::on_resetTimescaleButton_clicked()
{
    ui->TimescaleSlider->setValue(10);
}

