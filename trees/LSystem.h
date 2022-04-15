#ifndef LSYSTEM_H
#define LSYSTEM_H

#include <map>
#include <string>
#include "Random.h"
#include <float.h>
#include <memory>

struct OutputProbability {
    std::string output;
    float probability;
    OutputProbability(std::string output, float probability) :
        output(output),
        probability(probability)
    {
    }
};

typedef std::vector<OutputProbability> OutputDistribution;

class LSystem
{
public:
    LSystem();
    LSystem(std::string axiom);
    void setAxiom(std::string axiom);
    void addRule(char input, OutputDistribution outputs);
    std::string applyRules(int iterations);

private:
    std::string m_axiom;
    std::map<char, OutputDistribution> m_rules;
    std::string sampleOutputDistribution(OutputDistribution outputDistribution);

};

#endif // LSYSTEM_H
