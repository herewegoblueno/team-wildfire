#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <QMainWindow>
#include <vector>

class SupportCanvas3D;
class ShaderEvolutionManager;
class ShaderEvolutionTestingScene;

namespace Ui {
    class MainWindow;
}

/**
 * @class MainWindow
 *
 * The main graphical user interface class (GUI class) for our application.
 */
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void openXmlFileForForestScene(QString file);
    void onSupportCanvasInitialized();
    void notifyFrameCompleted();
    void updateModuleSelectionOptions(std::vector<int> moduleIDs);

protected:

    // Overridden from QWidget. Handles the window close event.
    virtual void closeEvent(QCloseEvent *e);
    void signalSettingsChanged(); //We could have done this with signals and slots; not a priority now

private slots:
    void toggleSeeBranchModules();

    void on_useLightingForShaders_stateChanged(int arg1);

    void on_mainTabWidget_currentChanged(int index);

    void changeCameraSettings(bool useOrbiting);

    void on_vizualizeForestVoxelGrid_stateChanged(int arg1);

    void on_useSmoke_stateChanged(int arg1);

    void on_useOrbitingCameraFire_stateChanged(int arg1);

    void on_forestVisualizationEyeXSlider_valueChanged(int value);

    void on_forestVisualizationEyeYSlider_valueChanged(int value);

    void on_forestVisualizationEyeZSlider_valueChanged(int value);

    void on_forestVisualizationEyeRSlider_valueChanged(int value);

    void on_visualizationTemperatureRangeSlider_valuesChanged(int , int );

    void on_VoxelVisOptionsDropbox_currentIndexChanged(int index);

    void on_FieldVisOptionsDropbox_currentIndexChanged(int index);

    void on_vizualizeForestField_stateChanged(int arg1);

    void on_TimescaleSlider_valueChanged(int value);

    void on_resetTimescaleButton_clicked();

    void on_ModuleSelectionDropDown_currentTextChanged(const QString &arg1);

    void on_viewOnlyModuleVoxelCheckbox_stateChanged(int arg1);

    void on_useOrbitingCamera_stateChanged(int arg1);

    void on_ModuleVisModeParentOptions_currentIndexChanged(int index);

private:
    Ui::MainWindow *ui;  // Auto-generated by Qt. DO NOT RENAME!

    SupportCanvas3D *m_canvas3D;
    void initializeFrameCounting();

    //For tracking fps
    int numberOfFramesRecorded;
    int timeWhenStartedCountingFPS;
    int timeAtLastFPSCounterUpdate;
};

#endif // MAINWINDOW_H
