#include "HemispherePlane.h"
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"
#include "GL/glew.h"


HemispherePlane::HemispherePlane(float radius, int columns, int rows):
    Surface(),
    radius(std::max(0.5f, radius)),
    columns(std::max(3, columns)),
    rows(std::max(1, rows))
{

}

HemispherePlane::~HemispherePlane(){

}

void HemispherePlane::createTriangles(){
    float angleChangePerRow = M_PI_2 / rows;
    float angleChangePerCol = 2 * M_PI / columns;
    int vertexesPerSide = rows + 1;

    //Making first line of points
    for(int r = 0; r <= rows; r++){
        float phi = angleChangePerRow * r;
        float z = radius * sin(phi);
        float y = radius * cos(phi);
        glm::vec3 point = glm::vec3(0, y, z);
        vertexBank.push_back({point, point});
    }

    for(int col = 0; col < columns; col++){
        //Make another line
        for(int r = 0; r <= rows; r++){
            float phi = angleChangePerRow * r;
            float z = radius * sin(phi);
            float y = radius * cos(phi);
            glm::vec3 point = glm::vec3(0, y, z);
            vertexBank.push_back({point, point});
        }

        //Rotate the new line of pints
        rotateVertexCollection(vertexBank, vertexesPerSide * (col + 1), vertexBank.size(), angleChangePerCol * (col + 1), glm::vec3(0, 1, 0));

        //Now let's connect the old line with the new line
        for(int r = 0; r < rows; r++){
            int beginning = vertexesPerSide * col + r;
            if (r == 0){ //The top row actually only comprises of a single triange, the rest will comprise of two
                triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
            }else{
                triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
                triangleBank.push_back({beginning + vertexesPerSide, beginning, beginning + 1 + vertexesPerSide});
            }
        }
    }
}
