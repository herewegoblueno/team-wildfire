#ifndef SHADERCODEDISPLAYER_H
#define SHADERCODEDISPLAYER_H

#include <QDialog>
#include "mainwindow.h"

class MainWindow;

//This class assumes that the parent is going to be a MainWindow
//Might not be best design but meeeh its always gonna be true
namespace Ui {
class ShaderCodeDisplayer;
}

class ShaderCodeDisplayer : public QDialog
{
    Q_OBJECT

public:
    explicit ShaderCodeDisplayer(QWidget *parent = nullptr);
    ~ShaderCodeDisplayer();
    void reset();
    void setShaderIndex(int i);

private slots:
    void on_closeButton_clicked();

    void on_updateButton_clicked();

    void on_decreaseButton_clicked();

    void on_increaseButton_clicked();

    void on_showGenerations_stateChanged(int arg1);

private:
    Ui::ShaderCodeDisplayer *ui;
    int currentIndex;
    MainWindow *mainWindow;
    bool showGenerations;

};

#endif // SHADERCODEDISPLAYER_H
