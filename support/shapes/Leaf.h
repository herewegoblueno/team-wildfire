#ifndef LEAF_H
#define LEAF_H

#include "Shape.h"
#include "Tessellator.h"

const int vertexRows = 3;
const int vertexCols = 5;

class Leaf : public Shape
{
public:
    Leaf();

private:
    void initializeVertexData() override;
    std::unique_ptr<Tessellator> m_tessellator;
    std::vector<glm::vec3> m_frontVertexGrid;
    std::vector<glm::vec3> m_backVertexGrid;
    void makeFrontVertexGrid();
    void makeBackVertexGrid();
};

#endif // LEAF_H
