#ifndef CIRCLEBASE_H
#define CIRCLEBASE_H

#define PI 3.14159265f

#include "Tessellator.h"
#include "TriMesh.h"
#include "Shape.h"

/** A top or bottom circle base, used for Cylinder and Cone */
class CircleBase : public Shape
{
public:
    CircleBase(int param1, int param2, bool isTopBase);
    ~CircleBase();
    std::vector<float> getVertexData();
    std::vector<glm::vec3> getVertices();
    std::vector<Triangle> getFaces();

private:
    void initializeVertexData() override;
    std::unique_ptr<Tessellator> m_tessellator;
    std::vector<glm::vec3> m_vertices;
    std::vector<Triangle> m_faces;
    std::vector<glm::vec3> makeVertexGrid();
    // Whether this is a top or bottom circle base
    bool m_isTopBase;
};


#endif // CIRCLEBASE_H
