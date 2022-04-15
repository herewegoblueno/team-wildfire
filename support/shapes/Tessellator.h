#ifndef TESSELLATOR_H
#define TESSELLATOR_H

#include "OpenGLShape.h"
#include "TriMesh.h"

/** Support class for tessellating a variety of shapes */
class Tessellator
{
public:
    Tessellator();
    ~Tessellator();
    std::vector<Triangle> tessellate(int gridWidth, int gridHeight);
    std::vector<float> processTriMesh(TriMesh triMesh);
    void setUncurvedMeshNormals(TriMesh triMesh);
};

#endif // TESSELLATOR_H
