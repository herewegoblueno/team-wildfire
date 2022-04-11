#ifndef NODEDISPENSER_H
#define NODEDISPENSER_H

#include "AstNodes.h"
#include <random>

class NodeDispenser
{
public:
    static std::minstd_rand rng;

    static int numberOfLeavesPossible;
    static std::uniform_int_distribution<> leafDist;
    static std::unique_ptr<GenotypeNode> getLeafNode();

    static int numberOfOperatorsPossible;
    static std::uniform_int_distribution<> operatorDist;
    static std::unique_ptr<GenotypeNode> getOperationNode();

    static std::unique_ptr<GenotypeNode> copyTree(GenotypeNode * parent);
    static std::unique_ptr<GenotypeNode> copyNode(GenotypeNode * parent);
};

#endif // NODEDISPENSER_H
