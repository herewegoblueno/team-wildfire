#include "Loop.h"
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtx/transform.hpp"
#include "GL/glew.h"


Loop::Loop(float innerHeight, float radius, float angleOfTip, int columns, int strips):
    Surface(),
    innerHeight(std::max(0.5f, innerHeight)),
    radius(std::max(0.1f, radius)),
    angleOfTip(std::max(0.005f, std::min(angleOfTip, (float)M_PI_2))),
    columns(std::max(3, columns)),
    strips(std::max(1, strips))
{

}

Loop::~Loop(){

}

void Loop::createTriangles(){
    int vertexesPerSide = strips + 1;
    int vertexesPerCircle = columns + 1;
    float angleBetweenCirclePoints = 2 * M_PI/columns;
    int circleVertexPairOffset = vertexesPerCircle - 1;

    //Make first circle
    //Then rotating this circle to amke the angle of tip, then translating it upwards
    glm::mat4 rotMat = glm::rotate((float)M_PI_2 - angleOfTip, glm::vec3(0, 0, 1));
    for(int col = 0; col <= columns; col++){
        float x = radius * cos(angleBetweenCirclePoints * col);
        float z = radius * sin(angleBetweenCirclePoints * col);
        glm::vec3 point = glm::vec3(x, 0, z);
        vertexBank.push_back({glm::vec3(rotMat * glm::vec4(point, 1.0)), point});
    }
    translateVertexCollection(vertexBank, 0, vertexBank.size(), glm::vec3(0, innerHeight/2, 0));

    //Doing something to get the bottom circle
    rotMat = glm::rotate( - ((float)M_PI_2 - angleOfTip), glm::vec3(0, 0, 1));
    for(int col = 0; col <= columns; col++){
        float x = radius * cos(angleBetweenCirclePoints * col);
        float z = radius * sin(angleBetweenCirclePoints * col);
        glm::vec3 point = glm::vec3(x, 0, z);
        vertexBank.push_back({glm::vec3(rotMat * glm::vec4(point, 1.0)), point});
    }
    translateVertexCollection(vertexBank, vertexesPerCircle, vertexBank.size(), glm::vec3(0, -innerHeight/2, 0));


    //Making first line of points connecting the two circles
    Vertex upperVertex = vertexBank[circleVertexPairOffset];
    Vertex lowerVertex = vertexBank[vertexesPerCircle + circleVertexPairOffset];
    glm::vec3 difference = lowerVertex.position - upperVertex.position; //from Upper to Lower
    for(int row = 0; row <= strips; row++){
        vertexBank.push_back({upperVertex.position + difference * (row / (float)strips), upperVertex.normal});
    }
    circleVertexPairOffset--;

    for(int col = 0; col < columns; col++){
        upperVertex = vertexBank[circleVertexPairOffset];
        lowerVertex = vertexBank[vertexesPerCircle + circleVertexPairOffset];
        difference = lowerVertex.position - upperVertex.position; //from Upper to Lower
        for(int row = 0; row <= strips; row++){
            vertexBank.push_back({upperVertex.position + difference * (row / (float)strips), upperVertex.normal});
        }
        circleVertexPairOffset--;

        //Now let's connect the old line with the new line
        for(int row = 0; row < strips; row++){
            int beginning = vertexesPerSide * col + row + (2 * vertexesPerCircle);
            triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
            triangleBank.push_back({beginning + vertexesPerSide, beginning, beginning + 1 + vertexesPerSide});
        }
    }
}
