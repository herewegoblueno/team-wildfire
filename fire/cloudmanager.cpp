#include "support/shapes/OpenGLShape.h"
#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/FBO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"
#include "support/gl/shaders/CS123Shader.h"


#include "support/lib/ResourceLoader.h"
#include <glm/gtx/transform.hpp>

#include "cloudmanager.h"
#include "voxels/voxelgrid.h"
#include "simulation/physics.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

CloudManager::CloudManager(VoxelGrid *grid) :
    m_grid(grid)
{
    m_fires.clear();
    std::vector<float> particle_quad = { -1, 1, 0, 0, 0,
                               -1, -1, 0, 0, 1,
                               1, 1, 0, 1, 0,
                                1, -1, 0, 1, 1};

    m_quad = std::make_unique<OpenGLShape>();
    m_quad->m_vertexData = particle_quad;
    std::vector<CS123::GL::VBOAttribMarker> attribs;
    attribs.push_back(VBOAttribMarker(ShaderAttrib::POSITION, 3, 0, VBOAttribMarker::DATA_TYPE::FLOAT, false));
    attribs.push_back(VBOAttribMarker(ShaderAttrib::TEXCOORD0, 2, 3*sizeof(GLfloat), VBOAttribMarker::DATA_TYPE::FLOAT, false));
    CS123::GL::VBO vbo(particle_quad.data(), 20, attribs, VBO::GEOMETRY_LAYOUT::LAYOUT_TRIANGLE_STRIP);
    m_quad->m_VAO = std::make_unique<VAO>(vbo, 4);

    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/cloud.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/cloud.frag");
    m_shader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);

    dist = std::normal_distribution<float>(mean, stddev);
}


void CloudManager::setCamera(glm::mat4 projection, glm::mat4 view)
{
    p = projection;
    v = view;
}

void CloudManager::setScale(float s)
{
    scale = s;
}

void CloudManager::draw()
{
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    m_shader->bind();

    m_shader->setUniform("scale", scale);
    m_shader->setUniform("p", p);
    m_shader->setUniform("v", v);

    int resolution = m_grid->getResolution();
    m_quad->bindVAO();
    int half_res = resolution/2;
    for(int x=0;x<resolution;x++)
        for(int y=resolution*0.7;y<resolution*0.95;y++)
            for(int z=0;z<resolution;z++)
            {
                if((x-half_res)*(x-half_res)+(z-half_res)*(z-half_res)+3*(y-half_res)*(y-half_res)>half_res*half_res)
                    continue;

                Voxel* v = m_grid->getVoxel(x,y,z);
                if(v->getCurrentState()->humidity>0.9)
                {
                    glm::mat4 M_cloud = glm::translate(glm::mat4(), glm::vec3(v->centerInWorldSpace));
                    m_shader->setUniform("m", M_cloud);
                    m_shader->setUniform("humi", v->getCurrentState()->humidity);
                    m_shader->setUniform("q_c", v->getCurrentState()->q_c);
                    m_quad->drawVAO();
                }
            }

//// testing
//    for(int x=18;x<23;x+=2)
//        for(int y=40;y<43;y+=2)
//            for(int z=18;z<21;z+=2)
//    {
//        Voxel* v = m_grid->getVoxel(x,y,z);
//        glm::mat4 M_cloud = glm::translate(glm::mat4(), glm::vec3(v->centerInWorldSpace));
//        m_shader->setUniform("m", M_cloud);
//        m_shader->setUniform("humi", v->getCurrentState()->humidity);
//        m_shader->setUniform("q_c", 0.8f);
//        m_quad->drawVAO();

//    }

    m_quad->unbindVAO();


    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}





