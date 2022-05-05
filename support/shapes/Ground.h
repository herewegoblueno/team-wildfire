#ifndef GROUND_H
#define GROUND_H

#include "Shape.h"

class Ground : public Shape
{
public:
    Ground();
private:
    void initializeVertexData() override;
    void buildTop();
    void buildBottom();
};

#endif // GROUND_H
