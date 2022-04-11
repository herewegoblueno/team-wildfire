#ifndef MUTATIONFACTORY_H
#define MUTATIONFACTORY_H

#include "AstNodes.h"
#include <random>
#include "ShaderEvolutionManager.h"
#include "NodeDispenser.h"

void mutate(GenotypeNode *current, GenotypeNode *parent, int generationToMutate);

GenotypeNode *replaceWithRandomTree(GenotypeNode *current, GenotypeNode *parent, int currentGeneration);
void addLeafOffset(GenotypeNode *current, int currentGeneration);
GenotypeNode *changeOperator(GenotypeNode *current, GenotypeNode *parent, int currentGeneration);

std::unique_ptr<ShaderGenotype> createOffspring(GenotypeNode *parent1, GenotypeNode *parent2);

void mutateWithOtherParentGenes(GenotypeNode *node, GenotypeNode *nodeParent, std::vector<GenotypeNode*> &parentsGenes, std::uniform_int_distribution<> geneChooser);
void listTreeChildren(GenotypeNode* t, std::vector<GenotypeNode*> &pointerStore);


#endif // MUTATIONFACTORY_H
