#include "LSystem.h"
#include <iostream>

LSystem::LSystem()
{
    m_axiom = "";
}

LSystem::LSystem(std::string axiom)
{
    m_axiom = axiom;
}

/** Add a rewriting rule */
void LSystem::addRule(char input, OutputDistribution outputs) {
    m_rules[input] = outputs;
}

/** Set the axiom of the L-system */
void LSystem::setAxiom(std::string axiom) {
    m_axiom = axiom;
}

/** Apply the rewriting rules to the axiom a given number of times */
std::string LSystem::applyRules(int iterations) {
    std::string current = m_axiom;
    for (int i = 0; i < iterations; i++) {
        std::string newString = "";
        for (auto &ch : current) {
            if (m_rules.count(ch)) {
                newString += sampleOutputDistribution(m_rules[ch]);
            } else {
                newString += ch;
            }
        }
        current = newString;
    }
    return current;
}

/** Sample from a distrubtion of possible output strings */
std::string LSystem::sampleOutputDistribution(OutputDistribution outputs) {
    // Verify that distrubtion is valid, i.e. sums to 1
    float sum = 0;
    for (OutputProbability &outputProb : outputs) {
        sum += outputProb.probability;
    }
    if (abs(sum - 1.0f) > FLT_EPSILON) {
        std::cerr << "Error: Output distribution does not sum to 1" << std::endl;
        return "";
    }
    float min = 0.f;
    float max = 0.f;
    float sample = randomFloat();
    for (OutputProbability &outputProb : outputs) {
        max += outputProb.probability;
        if (sample >= min && sample <= max) {
            return outputProb.output;
        }
        min += outputProb.probability;
    }
    return outputs[0].output;
}
