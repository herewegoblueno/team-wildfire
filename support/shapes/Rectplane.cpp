#include "Rectplane.h"
#include <vector>

RectPlane::RectPlane(float width, float height, int cellsPerEdge):
    Surface(),
    width(width),
    height(height),
    cellsPerEdge(std::max(1, cellsPerEdge))
{

}

RectPlane::~RectPlane(){

}

void RectPlane::createTriangles(){
    float cellWidth = width/cellsPerEdge;
    float cellHeight = height/cellsPerEdge;
    int vertexesPerSide = cellsPerEdge + 1;

    //Making first line of points
    for(int r = 0; r <= cellsPerEdge; r++){
        glm::vec3 point = glm::vec3(0, 0, cellHeight * r);
        vertexBank.push_back({point, glm::vec3(0, 1.f, 0)});
    }

    for(int col = 0; col < cellsPerEdge; col++){
        //Make another line
        for(int r = 0; r <= cellsPerEdge; r++){
            glm::vec3 point = glm::vec3(0, 0, cellHeight * r);
            vertexBank.push_back({point, glm::vec3(0, 1.f, 0)});
        }

        //Translate the new line of points to the right
        translateVertexCollection(vertexBank, vertexesPerSide * (col + 1), vertexBank.size(), glm::vec3(cellWidth * (col + 1), 0, 0));

        //Now let's connect the old line with the new line
        for(int r = 0; r < cellsPerEdge; r++){
            int beginning = vertexesPerSide * col + r;
            triangleBank.push_back({beginning, beginning + 1, beginning + 1 + vertexesPerSide});
            triangleBank.push_back({beginning + vertexesPerSide, beginning, beginning + 1 + vertexesPerSide});
        }
    }
}
