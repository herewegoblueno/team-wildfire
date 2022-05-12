#include "support/shapes/OpenGLShape.h"
#include "support/gl/textures/Texture2D.h"
#include "support/gl/textures/TextureParametersBuilder.h"

#include "support/gl/datatype/VAO.h"
#include "support/gl/datatype/FBO.h"
#include "support/gl/datatype/VBO.h"
#include "support/gl/datatype/VBOAttribMarker.h"
#include "support/gl/shaders/ShaderAttribLocations.h"
#include "support/gl/shaders/CS123Shader.h"


#include "support/lib/ResourceLoader.h"
#include <glm/gtx/transform.hpp>

#include "firemanager.h"
#include "voxels/voxelgrid.h"
#include "simulation/physics.h"

#include <iostream>
#include <QImage>
#include <QGLWidget>


using namespace CS123::GL;

FireManager::FireManager(VoxelGrid *grid) :
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


    QImage img(":/textures/fire2.png");
    QImage gl_img = QGLWidget::convertToGLFormat(img);
    m_texture = std::make_unique<Texture2D>(gl_img.bits(), gl_img.width(), gl_img.height());

    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/particle.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/fire.frag");
    m_fireshader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);

    vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/particle.vert");
    fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/smoke.frag");
    m_smokeshader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}


void FireManager::addFire(Module *m, glm::vec3 pos, float size)
{
    int density = densityFromSize(size);
    std::shared_ptr<Fire> fire = std::make_shared<Fire>(density, pos, size, m_grid);
    m_fires.insert({m, fire});
}

void FireManager::setModuleFireSizes(Module *m, float size)
{
    auto equalRange = m_fires.equal_range(m);
    int count = std::distance(equalRange.first, equalRange.second);
    if (count == 0) {
        return;
    }
    for (auto it = equalRange.first; it != equalRange.second; it++) {
        auto fire = it->second;
        fire->setSize(size);
    }
}

void FireManager::removeFires(Module *m)
{
    m_fires.erase(m);
}

void FireManager::setScale(float fire_particle_size, float smoke_particle_size)
{
    scale_fire = fire_particle_size;
    scale_smoke = smoke_particle_size;
}

void FireManager::setCamera(glm::mat4 projection, glm::mat4 view)
{
    p = projection;
    v = view;
}



void FireManager::drawFires(float time, bool smoke)
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    m_fireshader->bind();
    // bind texture (though currently it's not even used in the shader since it doesn't work on Mac)
    TextureParametersBuilder builder;
    builder.setFilter(TextureParameters::FILTER_METHOD::LINEAR);
    builder.setWrap(TextureParameters::WRAP_METHOD::REPEAT);

    TextureParameters parameters = builder.build();
    parameters.applyTo(*m_texture);
    std::string filename = "sprite";
    m_fireshader->setTexture(filename, *m_texture);
    m_fireshader->setUniform("scale", scale_fire);
    m_fireshader->setUniform("p", p);
    m_fireshader->setUniform("v", v);

    m_quad->bindVAO();
    for(auto &fire : m_fires){
        fire.second->update_particles(time);
        fire.second->drawParticles(m_fireshader.get(), m_quad.get());
    }
    m_quad->unbindVAO();

    m_fireshader->unbind();

    if (smoke)
    {

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_smokeshader->bind();
        m_smokeshader->setUniform("scale", scale_smoke);
        m_smokeshader->setUniform("p", p);
        m_smokeshader->setUniform("v", v);
        m_quad->bindVAO();
        for(auto &fire:m_fires){
            fire.second->updateSmoke(time);
            fire.second->drawSmoke(m_fireshader.get(), m_quad.get());
        }
        m_quad->unbindVAO();
        m_smokeshader->unbind();
    }

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);
}


/** Return size of fire based on the mass change rate */
double FireManager::massChangeRateToFireSize(double massChangeRate) {
    double massLossRate = -massChangeRate;
    double minFireSize = 0.4;
    double maxFireSize = 1.0;
    double minMassLossRate = 0.1;
    double maxMassLossRate = 0.7;
    double massLossRateDiff = maxMassLossRate - minMassLossRate;
    if (massLossRate < minMassLossRate) return minFireSize;
    if (massLossRate > maxMassLossRate) return maxFireSize;
    double massLossRatio = (massLossRate - minMassLossRate) / massLossRateDiff;
    double normalizedFireSize = Module::sigmoidFunc(massLossRatio);
    return minFireSize + massLossRateDiff * normalizedFireSize;
}


