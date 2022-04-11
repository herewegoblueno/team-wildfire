#ifndef ASTNODES_H
#define ASTNODES_H

#include <string>
#include <vector>
#include <random>
#include <memory>

//Useful for assing the "similarity" of nodes
enum GenotypeNodeClassification{
    LEAF,
    ARITHMETIC, //Things like + and *, as well as things like averages and minmax
    NOISE,
    TRIGONOMETRIC,
    TIME_BOUND,
    UNARY_OPERATION,
    BINARY_OPERATION
};


//The base class that all genotype nodes will inherit from
class GenotypeNode {
public:
    GenotypeNode() {}; //Number of needed children will be 0 (for leaves)
    GenotypeNode(int no_of_children): numberOfChildrenNeeded(no_of_children) {}; //For operator nodes
    virtual ~GenotypeNode() {};

    std::string stringify(bool showGenAnnotations = false);
    //Making this an abstract class.
    //Subclasses that override this should keep it virtual to facilitate dynamic_cast<>
    virtual std::string stringifyDispatch(bool showAnnotations) = 0;

    int numberOfChildrenNeeded = 0;
    std::vector<std::unique_ptr<GenotypeNode>> children;
    int generation = 0; //The generation of the AST that this node came into existence

    virtual std::vector<GenotypeNodeClassification> getClassifications();
    bool containsClassification(GenotypeNodeClassification c);
};

class ShaderGenotype{
public:
    ShaderGenotype(std::unique_ptr<GenotypeNode> rt, int birthGeneration = 0);
    std::unique_ptr<GenotypeNode> root;
    int currentGeneration;
    int birthGeneration;
};




//******************LEAF NODES******************

//Leaf nodes will inheret from this
class XPositionNode : public GenotypeNode {
public:
    XPositionNode(float x, float y, float z);
   virtual std::string stringifyDispatch(bool a) override;
   std::vector<GenotypeNodeClassification> getClassifications() override;
   float offsetX; float offsetY; float offsetZ;
   XPositionNode(): offsetX(0), offsetY(0), offsetZ(0) {};
};

class YPositionNode : public GenotypeNode {
public:
    YPositionNode(float x, float y, float z);
   virtual std::string stringifyDispatch(bool a) override;
   std::vector<GenotypeNodeClassification> getClassifications() override;
   float offsetX; float offsetY; float offsetZ;
   YPositionNode(): offsetX(0), offsetY(0), offsetZ(0) {};
};

class ZPositionNode : public GenotypeNode {
public:
    ZPositionNode(float x, float y, float z);
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
    float offsetX; float offsetY; float offsetZ;
    ZPositionNode(): offsetX(0), offsetY(0), offsetZ(0) {};
};

class TimeNode : public GenotypeNode {
public:
    TimeNode(float x, float y, float z);
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
    float offsetX; float offsetY; float offsetZ;
    TimeNode(): offsetX(0), offsetY(0), offsetZ(0) {};
};

class RandomVecNode : public GenotypeNode {
public:
    RandomVecNode(int seed);
    RandomVecNode(int seed, int x, int y, int z);
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
    int m_seed;
    float offsetX; float offsetY; float offsetZ;

private:
    float min = -0.8;
    float max = 1.7;
};



//******************OPERATION NODES******************

class AdditionNode : public GenotypeNode {
public:
    AdditionNode() : GenotypeNode(2){}
    virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class SubtractionNode : public GenotypeNode {
public:
    SubtractionNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class DivisionNode : public GenotypeNode {
public:
    DivisionNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class MultiplicationNode : public GenotypeNode {
public:
    MultiplicationNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

//Not currenly in use: creates very noise shaders
//class ModulusNode : public GenotypeNode {
//public:
//    ModulusNode() : GenotypeNode(2){}
//   virtual std::string stringifyDispatch(bool a) override;
//    std::vector<GenotypeNodeClassification> getClassifications() override;
//};

class AbsoluteValueNode : public GenotypeNode {
public:
    AbsoluteValueNode() : GenotypeNode(1){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class CrossProductNode : public GenotypeNode {
public:
    CrossProductNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class SinNode : public GenotypeNode {
public:
    SinNode() : GenotypeNode(1){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class CosNode : public GenotypeNode {
public:
    CosNode() : GenotypeNode(1){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class AtanNode : public GenotypeNode {
public:
    AtanNode() : GenotypeNode(1){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class MaxNode : public GenotypeNode {
public:
    MaxNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class MinNode : public GenotypeNode {
public:
    MinNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class PerlinNoiseNode : public GenotypeNode {
public:
    PerlinNoiseNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class XTransplantNode : public GenotypeNode {
public:
    XTransplantNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class YTransplantNode : public GenotypeNode {
public:
    YTransplantNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class ZTransplantNode : public GenotypeNode {
public:
    ZTransplantNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class AverageNode : public GenotypeNode {
public:
    AverageNode() : GenotypeNode(2){}
   virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;
};

class JuliaFractalNode : public GenotypeNode {
public:
    JuliaFractalNode(int seed);
    virtual std::string stringifyDispatch(bool a) override;
    std::vector<GenotypeNodeClassification> getClassifications() override;

    int m_seed;
    bool usePositionForSeed;
    bool breakAfter;
    std::minstd_rand rng;

    static std::uniform_int_distribution<> binaryDist;
    static std::uniform_int_distribution<> layerChoiceDist;
    std::vector<int> layerChoicesAndComponents;
};

#endif // ASTNODES_H
