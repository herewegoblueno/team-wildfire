#ifndef LSYSTEM_H
#define LSYSTEM_H

#include "Turtle.h"
#include <string>
#include <map>
#include <memory>

class LSystem
{
public:
    LSystem();
    LSystem(const std::map<std::string, std::string> & mappings, const std::string start, const float angle);
    std::string generate(int replacements);
    std::vector<glm::vec3> getStartingPoints(void);
    std::vector<glm::vec3> getEndingPoints(void);
    std::vector<glm::vec3> getLeaves(void);
    void draw(void);
private:
    std::string m_current;
    std::map<std::string, std::string> m_mappings;
    float m_length;
    float m_angle;
    void replace(void);
    // need something to get cylinders from turtle
    std::unique_ptr<Turtle> m_turtle;


};

#endif // LSYSTEM_H
