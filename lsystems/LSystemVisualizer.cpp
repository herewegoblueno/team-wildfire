#include "LSystemVisualizer.h"
#include "glm/gtx/transform.hpp"
#include <iostream>
#include "support/Settings.h"


LSystemVisualizer::LSystemVisualizer()
{
    // test map for trees
    // std::map<std::string, std::string> alphabet;
    // alphabet.insert(std::pair<std::string, std::string>("F", "FF+[+F-F-F]-[-F+F+F]"));
    // make L System
    // m_LSystem = std::make_unique<LSystem>(alphabet, "F", M_PI/6.f);
    m_LSystem = std::make_unique<LSystem>(LSystemUtils::getMap(settings.lSystemType), LSystemUtils::getStart(settings.lSystemType), LSystemUtils::getAngle(settings.lSystemType));
    m_LSystem->generate(settings.numRecursions);
    m_LSystem->draw();
    // get coords
    m_startingPoints = m_LSystem->getStartingPoints();
    m_endingPoints = m_LSystem->getEndingPoints();
    m_leaves = m_LSystem->getLeaves();
    leaf_len = 0.15f;
}

int LSystemVisualizer::getNumCyls() {
    return m_startingPoints.size();
}

int LSystemVisualizer::getNumLeaves() {
    return m_leaves.size();
}

glm::mat4x4 LSystemVisualizer::getTransformationMatrix(int index) {
    glm::vec3 start = m_startingPoints.at(index);
    glm::vec3 end = m_endingPoints.at(index);
    // std::cout << "index " << index << " start " << start.y << " end " << end.y << std::endl;
    // std::cout << "index " << index << " x start " << start.x << " end " << end.x << std::endl;
    // std::cout << "index " << index << " z start " << start.z << " end " << end.z << std::endl;

    // position to translate to
    glm::vec3 pos = 0.5f * (start + end);
    // scaling - y direction, by length of vector
    glm::vec3 scale;
    scale.y = glm::length(end - start);
    scale.x = 0.025;
    scale.z = 0.025;

    // rotate to the direction vector
    glm::vec3 axis = glm::cross(glm::vec3(0, 1, 0), end - start);
    float sinangle = glm::length(axis)/(glm::length(end - start));
    float cosangle = glm::dot(glm::vec3(0, 1, 0), end - start)/(glm::length(end - start));
    axis = glm::normalize(axis);

    // case where no rotation needed
    if(abs(glm::normalize(end - start).y - 1.f) <= 0.0001) {
        return glm::translate(pos)*glm::scale(scale);
    }
    // edge case where sin domain makes some angles backwards
    if(acos(cosangle) > asin(sinangle)) {
        return glm::translate(pos)*glm::rotate((float)acos(cosangle), axis)*glm::scale(scale);
    }
    return glm::translate(pos)*glm::rotate((float)asin(sinangle), axis)*glm::scale(scale);

}

glm::mat4x4 LSystemVisualizer::getLeafMatrix(int index) {
    glm::vec3 start = m_leaves.at(index);
    // leaf points down in a random direction
    // generate random amount down
    // random number between 0 and 1
    float randY = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    // generate random x/z direction
    randY *= 0.75f;
    float randomX = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    randomX -= 0.5f;
    float randomZ = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    randomZ -= 0.5f;

    glm::vec3 leafDir = glm::normalize(glm::vec3(randomX, -randY, randomZ));

    // find perpendicular vector
    glm::vec3 normal = glm::cross(leafDir, leafDir + glm::vec3(0.5f, 0, 0));

    glm::vec3 axis = glm::cross(glm::vec3(0, 1, 0), normal);
    float sinangle = glm::length(axis)/(glm::length(normal));
    float cosangle = glm::dot(glm::vec3(0, 1, 0), normal)/(glm::length(normal));
    axis = glm::normalize(axis);


    glm::vec3 leafPos = start + leafDir*leaf_len*0.5f;
    // scale leaf len
    glm::vec3 scale = glm::vec3(leaf_len/2.f, 0.01f, leaf_len);

    // edge case where sin domain makes some angles backwards
    if(acos(cosangle) > asin(sinangle)) {
        return glm::translate(leafPos)*glm::rotate((float)acos(cosangle), axis)*glm::scale(scale);
    }
    return glm::translate(leafPos)*glm::rotate((float)asin(sinangle), axis)*glm::scale(scale);



}
