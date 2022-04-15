#ifndef FRUIT_H
#define FRUIT_H
#include "Sphere.h"

class Fruit : public Shape
{
public:
    Fruit(int param1, int param2);
private:
    std::unique_ptr<Sphere> m_sphere;
    void initializeVertexData() override;

};

#endif // FRUIT_H
