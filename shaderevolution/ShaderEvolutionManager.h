#ifndef SHADEREVOLUTIONMANAGER_H
#define SHADEREVOLUTIONMANAGER_H

#include "mainwindow.h"
#include "AstNodes.h"
#include <random>
#include "support/scenegraph/ShaderEvolutionTestingScene.h"


class MainWindow;

class ShaderEvolutionManager
{
public:
    void init(MainWindow * window);
    MainWindow * m_window;

    std::unique_ptr<GenotypeNode> generateTree(int chanceOfOperator);
    std::unique_ptr<GenotypeNode> generateTree();
    void setNodeGeneration(GenotypeNode *node, int generation);
    void setTreeGeneration(GenotypeNode *node, int generation);

    std::minstd_rand rng;
    std::uniform_int_distribution<> changeDist;
    int maxProbability;

    QString m_shaderScenePath;
    void initializeShaderScene();
    void mutateSelected(ShaderEvolutionTestingScene * scene);
    void refreshSelected(ShaderEvolutionTestingScene * scene);
    void replaceWithMutationsOfDonor(ShaderEvolutionTestingScene * scene);
    void replaceSelectedWithOffspring(ShaderEvolutionTestingScene * scene);


    std::vector<bool> shaderSelections;
    int donorShaderIndex;
    int parent1ShaderIndex;
    int parent2ShaderIndex;
};

// The global ShaderEvolutionManager object, initialized in
extern ShaderEvolutionManager SEManager;

#endif // SHADEREVOLUTION_H
