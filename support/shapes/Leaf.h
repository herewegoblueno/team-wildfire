#ifndef LEAF_H
#define LEAF_H

#include "Shape.h"

class Leaf : public Shape
{
public:
    Leaf();
private:
    void initializeVertexData() override;
    void buildFront();
    void buildBack();
};

#endif // LEAF_H
