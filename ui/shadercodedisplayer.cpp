#include "shadercodedisplayer.h"
#include "ui_shadercodedisplayer.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/scenegraph/ShaderEvolutionTestingScene.h"


ShaderCodeDisplayer::ShaderCodeDisplayer(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ShaderCodeDisplayer),
    mainWindow((MainWindow *) parent),
    showGenerations(false)
{
    ui->setupUi(this);
}

ShaderCodeDisplayer::~ShaderCodeDisplayer()
{
    delete ui;
}

void ShaderCodeDisplayer::reset()
{
    setShaderIndex(0);
}

void ShaderCodeDisplayer::setShaderIndex(int i)
{
    if (!mainWindow) return;
    currentIndex = i;
    ui->shaderIndex->setText(QString::number(i + 1));
    ui->shaderIndex->repaint();
    ui->shaderCode->setText(
                (!showGenerations ? "" : QString::fromStdString(
                                      "<b><i><font color='#0000ff'> Generation at birth: ["
                                      + std::to_string((*mainWindow->getShaderScene()->getShaderGenotypes())[i]->birthGeneration)
                                      + "]<br>Current Generation: ["
                                      + std::to_string((*mainWindow->getShaderScene()->getShaderGenotypes())[i]->currentGeneration)
                                      + "]</font></i></b><br>")) +
                QString::fromStdString(mainWindow->getShaderScene()->getShaderSource(i, showGenerations)));
    updateGeometry();
}

void ShaderCodeDisplayer::on_closeButton_clicked()
{
    this->close();
}

void ShaderCodeDisplayer::on_updateButton_clicked()
{
    setShaderIndex(currentIndex);
}

void ShaderCodeDisplayer::on_decreaseButton_clicked()
{
    currentIndex = (currentIndex - 1);
    if (currentIndex < 0) currentIndex = ShaderEvolutionTestingScene::numberOfTestShaders - 1;
    setShaderIndex(currentIndex);
}

void ShaderCodeDisplayer::on_increaseButton_clicked()
{
    currentIndex = (currentIndex + 1) % ShaderEvolutionTestingScene::numberOfTestShaders;
    setShaderIndex(currentIndex);
}

void ShaderCodeDisplayer::on_showGenerations_stateChanged(int state)
{
    showGenerations = state == Qt::CheckState::Checked;
    setShaderIndex(currentIndex);
}
