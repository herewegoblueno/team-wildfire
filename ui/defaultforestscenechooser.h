#ifndef DEFAULTFORESTSCENECHOOSER_H
#define DEFAULTFORESTSCENECHOOSER_H
#include <QDialog>
#include "mainwindow.h"

class MainWindow;

namespace Ui {
class DefaultForestSceneChooser;
}

class DefaultForestSceneChooser : public QDialog
{
    Q_OBJECT

public:
    explicit DefaultForestSceneChooser(QWidget *parent = nullptr);
    ~DefaultForestSceneChooser();

private slots:
    void on_comboBox_currentIndexChanged(int index);

private:
    Ui::DefaultForestSceneChooser *ui;
    MainWindow *mainWindow;
};

#endif // DEFAULTFORESTSCENECHOOSER_H
