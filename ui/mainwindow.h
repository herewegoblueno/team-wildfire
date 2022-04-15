#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <QMainWindow>

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

protected:

    // Overridden from QWidget. Handles the window close event.
    virtual void closeEvent(QCloseEvent *e);
    void signalSettingsChanged(); //We could have done this with signals and slots; not a priority now

private slots:

    void on_useLightingForShaders_stateChanged(int arg1);

    void on_mainTabWidget_currentChanged(int index);

    void changeCameraSettings(bool useOrbiting);

    void on_vizualizeForestVoxelGrid_stateChanged(int arg1);

private:
    Ui::MainWindow *ui;  // Auto-generated by Qt. DO NOT RENAME!

    SupportCanvas3D *m_canvas3D;
};

#endif // MAINWINDOW_H
