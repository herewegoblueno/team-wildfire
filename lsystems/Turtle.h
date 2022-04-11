#ifndef TURTLE_H
#define TURTLE_H
#include <glm/glm.hpp>
#include <vector>
#include <stack>

class Turtle
{
public:
    Turtle();
    void forward(float dist);
    void turn(float angle);
    void roll(float angle);
    void pitch(float angle);
    void turnAround(void);
    void push(void);
    void pop(void);
    std::vector<glm::vec3> getStartingCoords(void);
    std::vector<glm::vec3> getEndingCoords(void);
    std::vector<glm::vec3> getLeafCoords(void);
private:
    glm::vec3 m_turtlePos;
    glm::vec3 m_turtleDir;
    // store start and end coords of each branch
    std::vector<glm::vec3> m_startingCoords;
    std::vector<glm::vec3> m_endingCoords;
    // store position of the turtle
    std::stack<glm::vec3> m_coordStack;
    std::stack<glm::vec3> m_angleStack;
    // store where leaves are added
    std::vector<glm::vec3> m_leaves;
    void addLeaf(void);

};

#endif // TURTLE_H
