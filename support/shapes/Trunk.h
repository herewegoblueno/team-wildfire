#ifndef TRUNK_H
#define TRUNK_H

#define PI 3.14159265f

#include "Tessellator.h"
#include "TriMesh.h"
#include "Shape.h"
#include "CircleBase.h"
#include "trees/MeshGenerator.h"

/**
 * Tapered cylinder bounded by a 1x1x1 cube centered at the origin.
 * Taper amount based on branchDecay in MeshGenerator.h
 */
class Trunk : public Shape
{
public:
    Trunk(int param1, int param2);
    ~Trunk();

private:
    void initializeVertexData() override;
    float m_taperAmt;
    std::unique_ptr<Tessellator> m_tessellator;
    std::unique_ptr<CircleBase> m_top;
    std::unique_ptr<CircleBase> m_bottom;
    std::vector<glm::vec3> makeSideGrid();
    std::vector<glm::vec3> scaleTop(std::vector<glm::vec3> vertices);
    void setSideNormals(std::vector<Triangle> &faces);
};


#endif // TRUNK_H
