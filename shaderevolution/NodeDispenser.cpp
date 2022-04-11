#include "NodeDispenser.h"
#include <ctime>

std::minstd_rand NodeDispenser::rng(std::time(0));

int NodeDispenser::numberOfLeavesPossible = 5;
std::uniform_int_distribution<> NodeDispenser::leafDist(1, NodeDispenser::numberOfLeavesPossible);

int NodeDispenser::numberOfOperatorsPossible = 17;
std::uniform_int_distribution<> NodeDispenser::operatorDist(1, NodeDispenser::numberOfOperatorsPossible);

std::unique_ptr<GenotypeNode> NodeDispenser::getLeafNode(){
    int choice = NodeDispenser::leafDist(NodeDispenser::rng);

    if (choice == 1) return std::make_unique<XPositionNode>();
    if (choice == 2) return std::make_unique<ZPositionNode>();
    if (choice == 3) return std::make_unique<YPositionNode>();
    if (choice == 4) return std::make_unique<TimeNode>();
    if (choice == 5) return std::make_unique<RandomVecNode>(rng());
    return nullptr;
}


std::unique_ptr<GenotypeNode> NodeDispenser::getOperationNode(){
    int choice = NodeDispenser::operatorDist(NodeDispenser::rng);

    if (choice == 1) return std::make_unique<AdditionNode>();
    if (choice == 2) return std::make_unique<SubtractionNode>();
    if (choice == 3) return std::make_unique<CrossProductNode>();
    if (choice == 4) return std::make_unique<MultiplicationNode>();
    if (choice == 5) return std::make_unique<DivisionNode>();
    if (choice == 6) return std::make_unique<AbsoluteValueNode>();
    if (choice == 7) return std::make_unique<SinNode>();
    if (choice == 8) return std::make_unique<CosNode>();
    if (choice == 9) return std::make_unique<AtanNode>();
    if (choice == 10) return std::make_unique<MinNode>();
    if (choice == 11) return std::make_unique<MaxNode>();
    if (choice == 12) return std::make_unique<PerlinNoiseNode>();
    if (choice == 13) return std::make_unique<XTransplantNode>();
    if (choice == 14) return std::make_unique<YTransplantNode>();
    if (choice == 15) return std::make_unique<ZTransplantNode>();
    if (choice == 16) return std::make_unique<AverageNode>();
    if (choice == 17) return std::make_unique<JuliaFractalNode>(rng());

    return nullptr;
}


std::unique_ptr<GenotypeNode> _copynode(bool includeChildren, GenotypeNode * parent){
    GenotypeNode * copy;

    //leafnodes
    if (XPositionNode * p = dynamic_cast<XPositionNode*>(parent)) copy = new XPositionNode(p->offsetX, p->offsetY, p->offsetZ);
    else if (ZPositionNode *p = dynamic_cast<ZPositionNode*>(parent)) copy = new ZPositionNode(p->offsetX, p->offsetY, p->offsetZ);
    else if (YPositionNode *p = dynamic_cast<YPositionNode*>(parent)) copy = new YPositionNode(p->offsetX, p->offsetY, p->offsetZ);
    else if (TimeNode *p = dynamic_cast<TimeNode*>(parent)) copy = new TimeNode(p->offsetX, p->offsetY, p->offsetZ);
    else if (RandomVecNode *p = dynamic_cast<RandomVecNode*>(parent))copy = new RandomVecNode(p->m_seed, p->offsetX, p->offsetY, p->offsetZ);
    //Operation nodes
    else if (AdditionNode *p = dynamic_cast<AdditionNode*>(parent)) copy = new AdditionNode();
    else if (SubtractionNode *p = dynamic_cast<SubtractionNode*>(parent)) copy = new SubtractionNode();
    else if (CrossProductNode *p = dynamic_cast<CrossProductNode*>(parent)) copy = new CrossProductNode();
    else if (MultiplicationNode *p = dynamic_cast<MultiplicationNode*>(parent)) copy = new MultiplicationNode();
    else if (DivisionNode *p = dynamic_cast<DivisionNode*>(parent)) copy = new DivisionNode();
    else if (AbsoluteValueNode *p = dynamic_cast<AbsoluteValueNode*>(parent)) copy = new AbsoluteValueNode();
    else if (SinNode *p = dynamic_cast<SinNode*>(parent)) copy = new SinNode();
    else if (CosNode *p = dynamic_cast<CosNode*>(parent)) copy = new CosNode();
    else if (AtanNode *p = dynamic_cast<AtanNode*>(parent)) copy = new AtanNode();
    else if (MinNode *p = dynamic_cast<MinNode*>(parent)) copy = new MinNode();
    else if (MaxNode *p = dynamic_cast<MaxNode*>(parent)) copy = new MaxNode();
    else if (PerlinNoiseNode *p = dynamic_cast<PerlinNoiseNode*>(parent)) copy = new PerlinNoiseNode();
    else if (XTransplantNode *p = dynamic_cast<XTransplantNode*>(parent)) copy = new XTransplantNode();
    else if (YTransplantNode *p = dynamic_cast<YTransplantNode*>(parent)) copy = new YTransplantNode();
    else if (ZTransplantNode *p = dynamic_cast<ZTransplantNode*>(parent)) copy = new ZTransplantNode();
    else if (AverageNode *p = dynamic_cast<AverageNode*>(parent)) copy = new AverageNode();
    else if (JuliaFractalNode *p = dynamic_cast<JuliaFractalNode*>(parent)) copy = new JuliaFractalNode(p->m_seed);

    if (includeChildren){
        for (int i = 0; i < parent->numberOfChildrenNeeded; i ++){
            copy->children.push_back(_copynode(true, parent->children[i].get()));
        }
    }

    copy->generation = parent->generation;
    return std::unique_ptr<GenotypeNode>(copy);
}

std::unique_ptr<GenotypeNode> NodeDispenser::copyTree(GenotypeNode * parent){
    return _copynode(true, parent);
}

std::unique_ptr<GenotypeNode> NodeDispenser::copyNode(GenotypeNode * parent){
    return _copynode(false, parent);
}

