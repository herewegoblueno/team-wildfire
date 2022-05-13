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
            break;
        case 1:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/twoRegions.xml";
            break;
        case 2:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/twoRegionsCloser.xml";
            break;
        case 3:
            mainWindow->currentForestXMLScene = ":/xmlScenes/xmlScenes/lessForestCover.xml";
            break;
    }

    this->hide();
}

