#include "LSystem.h"
#include <iostream>
#include "support/Settings.h"
#include "LSystemUtils.h"
#include "time.h"

LSystem::LSystem()
{

}

// set up L system mappings
LSystem::LSystem(const std::map<std::string, std::string> & mappings, const std::string start, const float angle) {
    m_mappings = mappings;
    m_current = start;
    m_turtle = std::make_unique<Turtle>();
    m_length = LSystemUtils::getStartingLength(settings.lSystemType);
    m_angle = angle;
    srand((unsigned)time(NULL));


}

// generate an L system string based off number of times to replace
std::string LSystem::generate(int replacements) {
    for(int i = 0; i < replacements; i++) {
        m_length *= 0.5f;
        replace();
    }
    // std::cout << m_current << std::endl;

    return m_current;
}

// given the current string, iterate through and draw with the turtle
void LSystem::draw(void) {
    // std::cout << "we finna draw!" << std::endl;
    int currentLen = m_current.size();
    for(int i = 0; i < currentLen; i++) {
        float angle = m_angle;
        if(settings.angleStochasticity) {
            // modify angle by random amount
            // random number between 0 and 1
            float randomNum = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            // random offset between -angle/2 and angle/2
            randomNum -= 0.5f;
            randomNum *= m_angle;
            angle = m_angle + randomNum;
        }
        switch(m_current[i]){
        case 'F': {
            //
            if(settings.lengthStochasticity) {
                // random number between 0 and 1
                float randomNum = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                randomNum -= 0.5f;
                randomNum *= m_length;
                if(randomNum + m_length > 0) {
                    m_turtle->forward(m_length + randomNum);
                } else {
                    m_turtle->forward(m_length);
                }

            } else {
                m_turtle->forward(m_length);
            }
            break;
        }
        case '+': {
            // turn left
            m_turtle->turn(angle);
            break;
        }
        case '-': {
            // turn right
            m_turtle->turn(-angle);
            break;
        }
        case '\\': {
            // roll left
            m_turtle->roll(angle);
            break;
        }
        case '/': {
            // roll right
            m_turtle->roll(-angle);
            break;
        }
        case '^': {
            // pitch up
            m_turtle->pitch(angle);
            break;
        }
        case '&': {
            // pitch down
            m_turtle->pitch(-angle);
            break;
        }
        case '[': {
            // push turtle position
            m_turtle->push();
            break;
        }
        case ']': {
            // push turtle position
            m_turtle->pop();
            break;
        }
        default: {
            // do nothing
        }
        }
    }
}

void LSystem::replace() {
    std::string replaced;
    int currentLen = m_current.size();
    for(int i = 0; i < currentLen; i++) {
        // check if there is a mapping from current character in string to another string
        if(m_mappings.find(m_current.substr(i, 1)) != m_mappings.end()) {
            // if mapping from current character found, append mapped value to new string
            replaced.append(m_mappings[m_current.substr(i, 1)]);
        } else {
            // else, add current character to string
            replaced.append(m_current.substr(i, 1));
        }
    }
    // set current to updated string
    m_current = replaced;

}

std::vector<glm::vec3> LSystem::getStartingPoints() {
    return m_turtle->getStartingCoords();
}

std::vector<glm::vec3> LSystem::getEndingPoints() {
    return m_turtle->getEndingCoords();
}

std::vector<glm::vec3> LSystem::getLeaves() {
    return m_turtle->getLeafCoords();
}
