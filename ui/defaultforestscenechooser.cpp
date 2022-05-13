#include "defaultforestscenechooser.h"
#include "ui_defaultforestscenechooser.h"
#include "support/Settings.h"

DefaultForestSceneChooser::DefaultForestSceneChooser(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DefaultForestSceneChooser),
    mainWindow((MainWindow *)parent)
{
    ui->setupUi(this);
}

DefaultForestSceneChooser::~DefaultForestSceneChooser()
{
    delete ui;
}

void DefaultForestSceneChooser::on_comboBox_currentIndexChanged(int index)
{
    switch(index){
        case 0:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/basicScene.xml";
        case 1:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/twoRegions.xml";
        case 2:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/twoRegionsCloser.xml";
        case 3:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/lessForestCover.xml";
    }

    this->hide();
}

