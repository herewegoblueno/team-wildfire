#include "MutationFactory.h"
#include <stdexcept>
#include <iostream>
#include "NodeDispenser.h"


std::minstd_rand RNG(std::time(0) + 100);



//For determining mutation chances
std::uniform_int_distribution<> mutationDist(1, 100);

//For determining offset amounts
std::uniform_real_distribution<> offsetDist(-2.f, 2.f);

//Useful for making yes or no choices
std::uniform_int_distribution<> binaryDist(0, 1);


//Mutation Strategy: Traverse though the tree and mutate things as you go
//As you do this, though, you might encounter nodes that are new to the tree due to
//the mutations you just did on the parents
//This is why you do generation checks.
void mutate(GenotypeNode *current, GenotypeNode *parent, int maxGenToMutate){
    if (current->generation > maxGenToMutate) return; //Skipped

//The next two chunks of commented out code are two different ways to mutate the genotype apart form
//Just adding offsets to the leaf nodes. The second one (changeOperator) is more promising as it
//causes less dramatic changes compared to the first one. They are commented out becuase the addLeafOffset
//is good enough for demonstrations - it causes a good enough range of changes

//    //Mutate the current node (if it's not a root node)
//    if (parent != nullptr){
//        if (mutationDist(RNG) < 10){
//            //10% chance for mutation
//            current = replaceWithRandomTree(current, parent, maxGenToMutate);
//        }
//    }

//    if (parent != nullptr){
//        if (!current->containsClassification(LEAF)){
//            if (mutationDist(RNG) < 10){
//                //10% chance for mutation
//                current = changeOperator(current, parent, maxGenToMutate);
//            }
//        }
//    }

    if (current->containsClassification(LEAF)){
        if (mutationDist(RNG) < 10){
            //10% chance for offset mutation
            addLeafOffset(current, maxGenToMutate);
        }
    }


    //Mutate the children
    for (int i = 0; i < current->numberOfChildrenNeeded; i ++){
       mutate(current->children[i].get(), current, maxGenToMutate);
    }
}

//Returns a pointer to the new tree that replaced current
GenotypeNode *replaceWithRandomTree(GenotypeNode *current, GenotypeNode *parent, int currentGeneration){
    std::unique_ptr<GenotypeNode> newTree = SEManager.generateTree(55);
    SEManager.setTreeGeneration(newTree.get(), currentGeneration + 1);

    //Let's find the index in the parent's children list that points to the to-be-replaced node
    for (int i = 0; i < parent->numberOfChildrenNeeded; i ++){
        if (parent->children[i].get() == current){
            parent->children[i] = std::move(newTree);         
            return parent->children[i].get();
        }
    }

    return nullptr;
}

//"For example (abs X) might become (cos X). If this mutation occurs,
//the arguments of the function are also adjusted if necessary to the correct number and types."
//(https://www.karlsims.com/papers/siggraph91.html)
GenotypeNode *changeOperator(GenotypeNode *current, GenotypeNode *parent, int currentGeneration){
    std::unique_ptr<GenotypeNode> newTree = NodeDispenser::getOperationNode();

    SEManager.setNodeGeneration(newTree.get(), currentGeneration + 1);

    int availableChildren = current->numberOfChildrenNeeded;
    //Tranfer all children from old parent to new parent
    int i;
    for (i = 0; i < availableChildren; i ++){
        if (i == newTree->numberOfChildrenNeeded) break; //The old parent gave us everything we need
        newTree->children.push_back(std::move(current->children[i]));
    }

    //If we need more children (so the old parent didn't have enough to give us...)
    //We'll recycle i
    for (; i < newTree->numberOfChildrenNeeded; i++){
        newTree->children.push_back(SEManager.generateTree(50));
        //These new children will need new generations
        SEManager.setTreeGeneration(newTree->children.back().get(), currentGeneration + 1);
    }

    //Let's find the index in the parent's children list that points to the to-be-replaced node
    for (int j = 0; j < parent->numberOfChildrenNeeded; j ++){
        if (parent->children[j].get() == current){
            parent->children[j] = std::move(newTree);
            return parent->children[j].get();
        }
    }

    return nullptr;
}



//This could have been written better but meh it gets the job done without adding
//more member functions to the nodes (or macros)
void addLeafOffset(GenotypeNode *node, int currentGeneration){
    if (XPositionNode * p = dynamic_cast<XPositionNode*>(node) ) {
       p->offsetX = offsetDist(RNG);
       p->offsetY = offsetDist(RNG);
       p->offsetZ = offsetDist(RNG);
    }
    else if (YPositionNode *p = dynamic_cast<YPositionNode*>(node) ) {
        p->offsetX = offsetDist(RNG);
        p->offsetY = offsetDist(RNG);
        p->offsetZ = offsetDist(RNG);
    }
    else if (ZPositionNode *p = dynamic_cast<ZPositionNode*>(node) ) {
        p->offsetX = offsetDist(RNG);
        p->offsetY = offsetDist(RNG);
        p->offsetZ = offsetDist(RNG);
     }
    else if (TimeNode *p = dynamic_cast<TimeNode*>(node) ) {
        p->offsetX = offsetDist(RNG);
        p->offsetY = offsetDist(RNG);
        p->offsetZ = offsetDist(RNG);
     }
    else if (RandomVecNode *p = dynamic_cast<RandomVecNode*>(node) ) {
        p->offsetX = offsetDist(RNG);
        p->offsetY = offsetDist(RNG);
        p->offsetZ = offsetDist(RNG);
     }
    else {
       throw std::invalid_argument("addLeafOffset: Was given a non-leaf node to add offset to!");
    }
    node->generation = currentGeneration + 1;
}


//New child will be created with generation 1
//All genes from primary parent will be tagged with generation 0
//All genes from secondary parent will be tagged with generation 1
std::unique_ptr<ShaderGenotype> createOffspring(GenotypeNode *parent1, GenotypeNode *parent2){
    GenotypeNode *primaryParent;
    GenotypeNode *secondaryParent;

    if (binaryDist(RNG) == 1){
        primaryParent = parent1;
        secondaryParent = parent2;
    }else{
        primaryParent = parent2;
        secondaryParent = parent1;
    }

    std::unique_ptr<GenotypeNode> child = NodeDispenser::copyTree(primaryParent);
    SEManager.setTreeGeneration(child.get(), 0);

    std::vector<GenotypeNode*> secondaryParentGenes;
    listTreeChildren(secondaryParent, secondaryParentGenes);
    std::uniform_int_distribution<> geneChooser(0, secondaryParentGenes.size() - 1);

    mutateWithOtherParentGenes(child.get(), nullptr, secondaryParentGenes, geneChooser);
    return std::make_unique<ShaderGenotype>(std::move(child), 1);
}



void listTreeChildren(GenotypeNode* t, std::vector<GenotypeNode*> &pointerStore){
    for (int i = 0; i < t->numberOfChildrenNeeded; i++){
        GenotypeNode* childPointer = t->children[i].get();
        pointerStore.push_back(childPointer);
        listTreeChildren(childPointer, pointerStore);
    }
}


//In this function, there's two notions of parent
//There's the node parent: the parent of the node in the gentypic tree structure
//And theres the Shader parent: the other shader genotype we're crossing genes with
void mutateWithOtherParentGenes(GenotypeNode *node,
                           GenotypeNode *nodeParent,
                           std::vector<GenotypeNode*> &parentsGenes,
                           std::uniform_int_distribution<> geneChooser){

    //We stop a branch of recursion the moment a substitiuion has occured
    //So we should never be doing this operation on a node that's from the other parent
    assert(node->generation == 0);

    //Potentially replace the current subtree with a subtree from the other parent's genepool
    //(if this isn't a root node)
    if (nodeParent != nullptr){
        if (mutationDist(RNG) < 30){ //30% chance for substitution

            //Make a copy of a random gene tree from the other parent's genepool
            std::unique_ptr<GenotypeNode> chosenGene = NodeDispenser::copyTree(parentsGenes[geneChooser(RNG)]);
            SEManager.setTreeGeneration(chosenGene.get(), 1);

            //Find the location in the current node's parent that links to it and replace it with chosenGene
            for (int i = 0; i < nodeParent->numberOfChildrenNeeded; i ++){
                if (nodeParent->children[i].get() != node) continue;
                nodeParent->children[i] = std::move(chosenGene);
                return;
            }
        }
    }

    //Continue Down the Tree if we haven't already mutated...
    for (int i = 0; i < node->numberOfChildrenNeeded; i ++){
       mutateWithOtherParentGenes(node->children[i].get(), node, parentsGenes, geneChooser);
    }
}


