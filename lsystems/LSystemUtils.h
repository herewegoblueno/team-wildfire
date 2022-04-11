#ifndef LSYSTEMUTILS_H
#define LSYSTEMUTILS_H

#include <map>
#include <string>
#include <math.h>

namespace LSystemUtils {

enum LSystemType {
    SEAWEED, MAPLE
};

const std::string seaweed_start = "F";
const std::map<std::string, std::string> seaweed_map {
    {std::string("F"), std::string("FF+[+F-F-F]-[-F+F+F]")}
};
const float seaweed_angle = M_PI/6.f;

const std::string maple_start = "X";
const std::map<std::string, std::string> maple_map {
    {std::string("X"), std::string("F[+X][-X]FX")},
    {std::string("F"), std::string("FF")}
};
const float maple_angle = M_PI/7.f;

const std::string natur_start = "X";
const std::map<std::string, std::string> natur_map {
    {std::string("X"), std::string("F-[[X]+X]+F[+FX]-X")},
    {std::string("F"), std::string("FF")}
};
const float natur_angle = M_PI/8.f;

const std::string bad_start = "F";
const std::map<std::string, std::string> bad_map {
    {std::string("F"), std::string("F[-EF[&&&A]]E[+F[^^^A]]")},
    {std::string("E"), std::string("F[  F[+++A]][^F[---A]]")}
};
const float bad_angle = M_PI/7.f;

const std::string bush_start = "A";
const std::map<std::string, std::string> bush_map {
    {std::string("A"), std::string("[&FL!A]/////`[&FL!A]///////`[&FL!A]")},
    {std::string("F"), std::string("S ///// F")},
    {std::string("S"), std::string("FL")}
    // ,
    // {std::string("L"), std::string("[```^^{-f+f+f-|-f+f+f}]")}
};
const float bush_angle = M_PI/8.f;

const std::string simple_start = "F";
const std::map<std::string, std::string> simple_map {
    {std::string("F"), std::string("F[-&^F][^++&F]|F[-&^F][+&F]")}
};
const float simple_angle = M_PI/12.f;

const std::string tree3D_start = "FA";
const std::map<std::string, std::string> tree3D_map {
    {std::string("A"), std::string("[&FLA^]////[&FLA]/////[^FLA]")},
    {std::string("F"), std::string("S ///// F")}
    // ,
    // {std::string("L"), std::string("[```^^{-f+f+f-|-f+f+f}]")}
};
const float tree3D_angle = M_PI/6.f;


std::string getStart(int type);
std::map<std::string, std::string> getMap(int type);
float getAngle(int type);
float getStartingLength(int type);


}

#endif // LSYSTEMUTILS_H
