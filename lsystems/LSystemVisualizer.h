#ifndef LSYSTEMVISUALIZER_H
#define LSYSTEMVISUALIZER_H

#include "LSystem.h"
#include "LSystemUtils.h"

class LSystemVisualizer
{
public:
    LSystemVisualizer();
    int getNumCyls(void);
    int getNumLeaves(void);
    glm::mat4x4 getTransformationMatrix(int index);
    glm::mat4x4 getLeafMatrix(int index);
private:
    std::unique_ptr<LSystem> m_LSystem;
    std::vector<glm::vec3> m_startingPoints;
    std::vector<glm::vec3> m_endingPoints;
    std::vector<glm::vec3> m_leaves;
    float leaf_len;
};

#endif // LSYSTEMVISUALIZER_H
