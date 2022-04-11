#ifndef SHADERCONSTRUCTOR_H
#define SHADERCONSTRUCTOR_H

#include <string>

class ShaderConstructor
{
public:
    static std::string genShader(std::string input);
private:
    static std::string beginning;
    static std::string end;
};

#endif // SHADERCONSTRUCTOR_H
