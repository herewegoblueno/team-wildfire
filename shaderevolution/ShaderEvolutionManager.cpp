#include "ShaderEvolutionManager.h"
#include <QMessageBox>
#include <QFileDialog>
#include <iostream>
#include "NodeDispenser.h"
#include "shaderevolution/MutationFactory.h"


ShaderEvolutionManager SEManager;

void ShaderEvolutionManager::init(MainWindow * window)
{
    m_window = window;
    maxProbability = 100;
    changeDist = std::uniform_int_distribution<>(1, maxProbability);
    shaderSelections.resize(ShaderEvolutionTestingScene::numberOfTestShaders);
    donorShaderIndex = 0;
    parent1ShaderIndex = 0;
    parent2ShaderIndex = 0;
}

void ShaderEvolutionManager::initializeShaderScene(){
    m_window->fileOpen(":/xmlScenes/xmlScenes/shaderTestingScene.xml");
}


std::unique_ptr<GenotypeNode> ShaderEvolutionManager::generateTree(){
    return generateTree(maxProbability);
}

//ChanceOfOperator [0, 100]
//This will make recursive calls that slowly reduce chanceOfOperator so that the
//basecase of leaf nodes is guaranteed to be met eventually
std::unique_ptr<GenotypeNode> ShaderEvolutionManager::generateTree(int chanceOfOperator){
    //Make choice
    int choice = changeDist(rng);
    std::unique_ptr<GenotypeNode> nodePointer;

    //Get node
    if (choice <= chanceOfOperator) nodePointer = NodeDispenser::getOperationNode();
    else nodePointer = NodeDispenser::getLeafNode();

    //Set up node
    for (int i = 0; i < nodePointer->numberOfChildrenNeeded; i++){
        nodePointer->children.push_back(generateTree(chanceOfOperator - 10));
    }
    return nodePointer;
}

//The following two shouldn't be needed really, since these should be set in the constructor of GenotypeNode
//But it'll be a hassle becuase I have lots of node types (and hence constructors to edit for each of them)
void ShaderEvolutionManager::setNodeGeneration(GenotypeNode *node, int generation){
    node->generation = generation;
}


void ShaderEvolutionManager::setTreeGeneration(GenotypeNode *node, int generation){
    node->generation = generation;
    for (int i = 0; i < node->numberOfChildrenNeeded; i++){
        setTreeGeneration(node->children[i].get(), generation);
    }
}


void ShaderEvolutionManager::mutateSelected(ShaderEvolutionTestingScene * scene){
    if (scene != nullptr){
        std::vector<std::unique_ptr<ShaderGenotype>>* vec = scene->getShaderGenotypes();

        for (int i = 0; i < ShaderEvolutionTestingScene::numberOfTestShaders; i ++){
            if (!shaderSelections[i]) continue;

            mutate((*vec)[i]->root.get(), nullptr, (*vec)[i]->currentGeneration);
            (*vec)[i]->currentGeneration ++;
        }
        scene->constructShaders();
    }
}


void ShaderEvolutionManager::refreshSelected(ShaderEvolutionTestingScene *scene){
    if (scene != nullptr){
        std::vector<std::unique_ptr<ShaderGenotype>>* vec = scene->getShaderGenotypes();

        for (int i = 0; i < ShaderEvolutionTestingScene::numberOfTestShaders; i ++){
            if (!shaderSelections[i]) continue;

            (*vec)[i] = std::make_unique<ShaderGenotype>(generateTree());
        }
        scene->constructShaders();
    }
}

void ShaderEvolutionManager::replaceWithMutationsOfDonor(ShaderEvolutionTestingScene *scene){
    if (scene != nullptr){
        std::vector<std::unique_ptr<ShaderGenotype>>* vec = scene->getShaderGenotypes();

        for (int i = 0; i < ShaderEvolutionTestingScene::numberOfTestShaders; i ++){
            if (!shaderSelections[i]) continue;

            if (i != donorShaderIndex){
                (*vec)[i] = std::make_unique<ShaderGenotype>(
                            NodeDispenser::copyTree((*vec)[donorShaderIndex]->root.get()),
                            (*vec)[donorShaderIndex]->currentGeneration);
            }

            mutate((*vec)[i]->root.get(), nullptr, (*vec)[i]->currentGeneration);
            (*vec)[i]->currentGeneration ++;

        }
        scene->constructShaders();
    }
}


void ShaderEvolutionManager::replaceSelectedWithOffspring(ShaderEvolutionTestingScene *scene){
    if (scene != nullptr){
        if (shaderSelections[parent1ShaderIndex] || shaderSelections[parent2ShaderIndex]){
            //Replacing a parent with its offspring could cause problems if you weren't done repalcing all other shaders with offspring
            QMessageBox::warning(m_window, "Not Safe!", "Sorry, but it's not memory safe to replace a parent with one of it's offspring");
            return;
        }

        std::vector<std::unique_ptr<ShaderGenotype>>* vec = scene->getShaderGenotypes();
        GenotypeNode *p1 = (*vec)[parent1ShaderIndex]->root.get();
        GenotypeNode *p2 = (*vec)[parent2ShaderIndex]->root.get();

        for (int i = 0; i < ShaderEvolutionTestingScene::numberOfTestShaders; i ++){
            if (!shaderSelections[i]) continue;
            (*vec)[i] = createOffspring(p1, p2);

        }
        scene->constructShaders();
    }
}


