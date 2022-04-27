#include "terrain.h"

#include <math.h>
#include "support/gl/shaders/ShaderAttribLocations.h"

Terrain::Terrain() : m_numRows(1000), m_numCols(m_numRows), m_isFilledIn(true)
{
}


/**
 * Returns a pseudo-random value between -1.0 and 1.0 for the given row and
 * column.
 */
float Terrain::randValue(int row, int col)
{
    // Gets good hill next to tree
    //return -1.0 + 2.0 * glm::fract(sin(row * 327.1f + col * 211.7f) * 23758.5453123f);

    return -1.0 + 2.0 * glm::fract(sin(row * 722.1f + col * 111.7f) * 76758.76123f);
}


/**
 * Returns the object-space position for the terrain vertex at the given row
 * and column.
 */
glm::vec3 Terrain::getPosition(int row, int col)
{
    glm::vec3 position;
    float size = m_numRows / scale;
    position.x = size * (2 * row/m_numRows - 1);
    position.y = 0;
    position.z = size * (2 * col/m_numCols - 1);

    // TODO: Adjust position.y using value noise.
    float new_row = glm::floor(row / 100.0f);
    float new_col = glm::floor(col / 100.0f);

    float row_weight = (row % 100) / 100.0f;
    float col_weight = (col % 100) / 100.0f;
    row_weight = 3*pow(row_weight, 2) - 2*pow(row_weight, 3);
    col_weight = 3*pow(col_weight, 2) - 2*pow(col_weight, 3);

    float r1 = glm::mix(randValue(new_row, new_col), randValue(new_row + 1, new_col), row_weight);
    float r2 = glm::mix(randValue(new_row, new_col + 1), randValue(new_row + 1, new_col + 1), row_weight);

    float y1 = 1.5f*glm::mix(r1, r2, col_weight);

    new_row = glm::floor(row / 10.0f);
    new_col = glm::floor(col / 10.0f);

    row_weight = (row % 10) / 10.0f;
    col_weight = (col % 10) / 10.0f;
    row_weight = 3*pow(row_weight, 2) - 2*pow(row_weight, 3);
    col_weight = 3*pow(col_weight, 2) - 2*pow(col_weight, 3);

    r1 = glm::mix(randValue(new_row, new_col), randValue(new_row + 1, new_col), row_weight);
    r2 = glm::mix(randValue(new_row, new_col + 1), randValue(new_row + 1, new_col + 1), row_weight);

    float y2 = glm::mix(r1, r2, col_weight) / 10.0;


    position.y = y1 + y2;// + y3 + y4 + y5 + y6;


    return position;
}


/**
 * Returns the normal vector for the terrain vertex at the given row and
 * column.
 */
glm::vec3 Terrain::getNormal(int row, int col)
{
    // TODO: Compute the normal at the given row and column using the positions
    //       of the neighboring vertices.

    std::vector<glm::vec3> points;
    glm::vec3 normal_sum = glm::vec3(0, 0, 0);

    glm::vec3 p1 = getPosition(row, col);

    points.reserve(8);
    for (int i = 1; i > -2; i--){
        points[1 - i] = getPosition(row - 1, col + i) - p1;
    }
    points[3] = getPosition(row, col - 1) - p1;
    for (int i = -1; i < 2; i++){
        points[5 + i] = getPosition(row + 1, col + i) - p1;
    }
    points[7] = getPosition(row, col + 1) - p1;

    for (int i = 0; i < 8; i++){
        normal_sum += glm::normalize(glm::cross(points[(i+1)%8], points[i]));
    }

    return glm::normalize(normal_sum);
}

bool Terrain::isFilledIn() {
    return m_isFilledIn;
}


float Terrain::getHeightFromWorld(glm::vec3 pos){

    float nearRow = (pos.x * scale / m_numRows + 1.f) / 2.f * m_numRows;
    float nearCol = (pos.z * scale / m_numRows + 1.f) / 2.f * m_numRows;

    int r1 = floor(nearRow);
    int r2 = ceil(nearRow);
    float rMix = nearRow - r1;

    int c1 = floor(nearCol);
    int c2 = ceil(nearCol);
    float cMix = nearCol - c1;

    float m1 = glm::mix(getPosition(r1, c1).y, getPosition(r2, c1).y, rMix);
    float m2 = glm::mix(getPosition(r1, c2).y, getPosition(r2, c2).y, rMix);

    return glm::mix(m1, m2, cMix);
    //return getPosition(r1, c1).y;
}

glm::vec3 Terrain::getNormalFromWorld(glm::vec3 pos){

    float nearRow = (pos.x * scale / m_numRows + 1.f) / 2.f * m_numRows;
    float nearCol = (pos.z * scale / m_numRows + 1.f) / 2.f * m_numRows;

    int r1 = floor(nearRow);
    int r2 = ceil(nearRow);
    float rMix = nearRow - r1;

    int c1 = floor(nearCol);
    int c2 = ceil(nearCol);
    float cMix = nearCol - c1;

    glm::vec3 m1 = glm::mix(getNormal(r1, c1), getNormal(r2, c1), rMix);
    glm::vec3 m2 = glm::mix(getNormal(r1, c2), getNormal(r2, c2), rMix);

    return glm::mix(m1, m2, cMix);
    //return getNormal(r1, c1);
}

/**
 * Initializes the terrain by storing positions and normals in a vertex buffer.
 */
std::vector<float> Terrain::init() {
    // Initializes a grid of vertices using triangle strips.
    int numVertices = (m_numRows - 1) * (2 * m_numCols + 2);
    glm::vec3 cur;
    std::vector<float> data(2 * numVertices * 3);
    int index = 0;
    for (int row = 0; row < m_numRows - 1; row++) {
        for (int col = m_numCols - 1; col >= 0; col--) {
            cur = getPosition(row, col);
            data[index++] = cur.x;
            data[index++] = cur.y;
            data[index++] = cur.z;

            cur = getNormal  (row, col);
            data[index++] = cur.x;
            data[index++] = cur.y;
            data[index++] = cur.z;

            cur = getPosition(row + 1, col);
            data[index++] = cur.x;
            data[index++] = cur.y;
            data[index++] = cur.z;

            cur = getNormal  (row + 1, col);
            data[index++] = cur.x;
            data[index++] = cur.y;
            data[index++] = cur.z;
        }
        cur = getPosition(row + 1, 0);
        data[index++] = cur.x;
        data[index++] = cur.y;
        data[index++] = cur.z;

        cur = getNormal  (row + 1, 0);
        data[index++] = cur.x;
        data[index++] = cur.y;
        data[index++] = cur.z;

        cur = getPosition(row + 1, m_numCols - 1);
        data[index++] = cur.x;
        data[index++] = cur.y;
        data[index++] = cur.z;

        cur = getNormal  (row + 1, m_numCols - 1);
        data[index++] = cur.x;
        data[index++] = cur.y;
        data[index++] = cur.z;
    }

    return data;
}


/**
 * Draws the terrain.
 */
void Terrain::draw()
{
    openGLShape->draw();
}
