#ifndef CUBE_H
#define CUBE_H

#include "Tessellator.h"
#include "TriMesh.h"
#include "Shape.h"

/** Cube with side length 1 centered at the origin */
class Cube : public Shape
{
public:
    Cube(int param1, int param2);
    ~Cube();

private:
    void initializeVertexData() override;
    std::unique_ptr<Tessellator> m_tessellator;
    std::vector<glm::vec3> makeSideGrid();
    std::vector<glm::vec3> makeTopGrid();
    std::vector<glm::vec3> makeBottomGrid();
};

#endif // CUBE_H
