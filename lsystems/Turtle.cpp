#include "Turtle.h"
// model turtle graphics, except using cylinders
#include "glm/gtx/transform.hpp"
#include "support/Settings.h"


Turtle::Turtle()
{
    // set up initial turtle position - bottom of unit shape
    m_turtlePos = glm::vec3(0, -1.25f, 0);
    // initially faces upwards
    m_turtleDir = glm::vec3(0, 1, 0);
    // initialize all vectors
    m_startingCoords = std::vector<glm::vec3>();
    m_endingCoords = std::vector<glm::vec3>();
    m_leaves = std::vector<glm::vec3>();
    // initialize the stacks
    m_coordStack = std::stack<glm::vec3>();
    m_angleStack = std::stack<glm::vec3>();


}

// move turtle forward by specified distance
void Turtle::forward(float dist) {
    // add new line to the stack
    m_startingCoords.push_back(m_turtlePos);
    // find new coords
    glm::vec3 newPos;
    newPos.x = m_turtlePos.x + m_turtleDir.x*dist;
    newPos.y = m_turtlePos.y + m_turtleDir.y*dist;
    newPos.z = m_turtlePos.z + m_turtleDir.z*dist;
    // update turtle position
    m_endingCoords.push_back(newPos);
    m_turtlePos = newPos;
}

// turn turtle left or right by given angle
void Turtle::turn(float angle) {
    // make a rotation matrix
    glm::mat4 rotation = glm::rotate(glm::mat4(1.f), angle, glm::vec3(1, 0, 0));
    // rotate direction vector using that matrix
    m_turtleDir = glm::normalize(glm::vec3(rotation*glm::vec4(m_turtleDir, 0)));


}

// roll turtle by a particular angle
void Turtle::roll(float angle) {
    // make a rotation matrix
    glm::mat4 rotation = glm::rotate(glm::mat4(1.f), angle, glm::vec3(0, 1, 0));
    // rotate direction vector using that matrix
    m_turtleDir = glm::normalize(glm::vec3(rotation*glm::vec4(m_turtleDir, 0)));
}

// pitch by a particular angle
void Turtle::pitch(float angle) {
    // make a rotation matrix
    glm::mat4 rotation = glm::rotate(glm::mat4(1.f), angle, glm::vec3(0, 0, 1));
    // rotate direction vector using that matrix
    m_turtleDir = glm::normalize(glm::vec3(rotation*glm::vec4(m_turtleDir, 0)));

}

// turn the turtle around
void Turtle::turnAround(void) {
    // negate the direction vector
    m_turtleDir = -m_turtleDir;

}

// save the turtle's current position
void Turtle::push(void) {
    m_coordStack.push(m_turtlePos);
    m_angleStack.push(m_turtleDir);
}

// restore the turtle to the last position on the stack
void Turtle::pop(void) {
    // if leaves are enabled, make a leaf here
    if(settings.hasLeaves) {
        addLeaf();
    }
    m_turtlePos = m_coordStack.top();
    m_coordStack.pop();
    m_turtleDir = m_angleStack.top();
    m_angleStack.pop();
}

std::vector<glm::vec3> Turtle::getStartingCoords() {
    return m_startingCoords;
}


std::vector<glm::vec3> Turtle::getEndingCoords() {
    return m_endingCoords;
}

void Turtle::addLeaf() {
    m_leaves.push_back(m_turtlePos);
}

std::vector<glm::vec3> Turtle::getLeafCoords() {
    return m_leaves;
}
