#include "CircularPlane.h"
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"
#include "GL/glew.h"


CircularPlane::CircularPlane(float radius, float height, int sectors, int strips):
    Surface(),
    radius(std::max(0.5f, radius)),
    height(std::max(0.f, height)),
    sectors(std::max(3, sectors)),
    strips(std::max(1, strips))
{

}

CircularPlane::~CircularPlane(){

}

void CircularPlane::createTriangles(){
    float angleDisplacementPerSector = (float)(M_PI * 2.f / (float)sectors);
    float hypotenuse = sqrt(radius * radius + height * height);
    int vertexesPerSide = strips + 1;

    Vertex topPoint = {glm::vec3(0, height, 0), glm::vec3(0, 1, 0)};
    //Aligning the top vertex's normal correctly, then putting the top point back
    rotateVertex(&topPoint, (M_PI / 2.f) - acos(height/hypotenuse), glm::vec3(1, 0, 0));
    topPoint.position = glm::vec3(0, height, 0);

    Vertex base = {glm::vec3(0, 0, radius), topPoint.normal};

    //Making first line of points
    glm::vec3 hypotenuseEqn = base.position - topPoint.position;
    for(int row = 0; row <= strips; row++){
        glm::vec3 point = topPoint.position + (hypotenuseEqn * (row/(float)strips));
        vertexBank.push_back({point, topPoint.normal});
    }

    for(int col = 0; col < sectors; col++){
        //Make another line
        for(int row = 0; row <= strips; row++){
            glm::vec3 point = topPoint.position + (hypotenuseEqn * (row/(float)strips));
            vertexBank.push_back({point, topPoint.normal});
        }

        //Rotate the new line of pints
        rotateVertexCollection(vertexBank, vertexesPerSide * (col + 1), vertexBank.size(), angleDisplacementPerSector * (col + 1), glm::vec3(0, 1, 0));

        //Now let's connect the old line with the new line
        for(int row = 0; row < strips; row++){
            int beginning = vertexesPerSide * col + row;
            if (row == 0){ //The top row actually only comprises of a single triange, the rest will comprise of two
                triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
            }else{
                triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
                triangleBank.push_back({beginning + vertexesPerSide, beginning, beginning + 1 + vertexesPerSide});
            }
        }
    }
}
