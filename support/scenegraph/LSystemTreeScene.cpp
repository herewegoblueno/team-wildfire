#include "LSystemTreeScene.h"
#include "GL/glew.h"
#include <QGLWidget>
#include "support/camera/Camera.h"

#include "support/Settings.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"

#include "support/shapes/Cube.h"
#include "support/shapes/Cone.h"
#include "support/shapes/Sphere.h"
#include "support/shapes/Cylinder.h"
#include "support/shapes/Torus.h"

#include <iostream>

using namespace CS123::GL;


LSystemTreeScene::LSystemTreeScene()
{
    loadPhongShader();

    shapeBank.resize(6);
    defineShapeBank();
    if(settings.numRecursions < 1) {
        settings.numRecursions = 1;
    }

    // make a new L System visualizer
    m_lSystemViz = std::make_unique<LSystemVisualizer>();
    makeLSystemVisualizer();
    // add some lights to the scene
    CS123SceneLightData light = {0, LightType::LIGHT_POINT, glm::vec4(1, 1, 1, 1), glm::vec3(0, 0, 1), glm::vec4(0, 1, 3, 0)};
    CS123SceneLightData light2 = {0, LightType::LIGHT_POINT, glm::vec4(1, 1, 1, 1), glm::vec3(0, 0, 1), glm::vec4(0, 1, -3, 0)};
    CS123SceneGlobalData globalLSys = {0.7f, 0.6f, 0.1f, 1};
    setGlobal(globalLSys);
    lightingInformation.push_back(light);
    lightingInformation.push_back(light2);
    setLights();
}

LSystemTreeScene::~LSystemTreeScene()
{
}

void LSystemTreeScene::loadPhongShader() {
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
    m_phongShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}

void LSystemTreeScene::render(SupportCanvas3D *context) {
    setClearColor(0.2,0.2,0.3,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //Phong pass
    m_phongShader->bind();
    setPhongSceneUniforms(context);
    setLights();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    renderGeometry();
    glBindTexture(GL_TEXTURE_2D, 0);
    m_phongShader->unbind();

}

void LSystemTreeScene::setPhongSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    m_phongShader->setUniform("useLighting", settings.useLighting);
    m_phongShader->setUniform("useArrowOffsets", false);
    m_phongShader->setUniform("p" , camera->getProjectionMatrix());
    m_phongShader->setUniform("v", camera->getViewMatrix());
}

void LSystemTreeScene::setMatrixUniforms(Shader *shader, SupportCanvas3D *context) {
    shader->setUniform("p", context->getCamera()->getProjectionMatrix());
    shader->setUniform("v", context->getCamera()->getViewMatrix());
}

void LSystemTreeScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        m_phongShader->setLight(lightingInformation[i]);
    }
}

void LSystemTreeScene::renderGeometry() {
    int size = primitives.size();
    for (int i = 0; i < size; i++){
        CS123ScenePrimitiveBundle bundle = primitives[i];
        CS123SceneMaterial mat = bundle.primitive.material;
        mat.cDiffuse *= globalData.kd;
        mat.cAmbient *= globalData.ka;
        mat.shininess *= globalData.ks;
        mat.cTransparent *= globalData.kt;

        m_phongShader->setUniform("m", bundle.model);
        m_phongShader->applyMaterial(mat);

        (shapeBank[(int) bundle.primitive.type])->draw();
    }
}

void LSystemTreeScene::settingsChanged() {
    // make a new LSystem with the current settings
    m_lSystemViz = std::make_unique<LSystemVisualizer>();
    makeLSystemVisualizer();
    renderGeometry();
}

void LSystemTreeScene::defineShapeBank(){
    //Can be linked to settings.parameter1-3, but since
    //we know the scenes that are being made we'll but hardcode it
    //Helpful for perfomance reasons too (prevents excessive tessellation)
    int p1 = 8;
    int p2 = 8;
    int p3 = 8;
    shapeBank[0] = std::make_unique<Cube>(p1);
    shapeBank[1] = std::make_unique<Cone>(p1, p2);
    shapeBank[2] = std::make_unique<Cylinder>(p1, p2);
    shapeBank[3] = std::make_unique<Torus>(p1, p2, p3);
    shapeBank[4] = std::make_unique<Sphere>(p1, p2);
    shapeBank[5] = std::make_unique<Sphere>(p1, p2); //Speres will substitite for meshes for now
}

void LSystemTreeScene::makeLSystemVisualizer() {
    int numCyls = m_lSystemViz->getNumCyls();
    CS123SceneMaterial material;
    material.clear();
    material.cAmbient.r = 0.13f;
    material.cAmbient.g = 0.1f;
    material.cAmbient.b = 0.05f;
    material.cDiffuse.r = 0.70f;
    material.cDiffuse.g = 0.5f;
    material.cDiffuse.b = 0.2f;

    material.cSpecular.r = material.cSpecular.g = material.cSpecular.b = 0.3;
    material.shininess = 12;
    // if seaweed, make material green
    if(settings.lSystemType == 1) {
        material.cAmbient.r = 0.04f;
        material.cAmbient.g = 0.2f;
        material.cAmbient.b = 0.04f;
        material.cDiffuse.r = 0.2f;
        material.cDiffuse.b = 0.4f;
    }
    primitives.clear();
    // add all cylinders to scene
    CS123ScenePrimitive cyl = {PrimitiveType::PRIMITIVE_CYLINDER, std::string(), material};
    for(int i = 0; i < numCyls; i++) {
        // make a new scene primitive
        addPrimitive(cyl, m_lSystemViz->getTransformationMatrix(i));

    }

    // if there are leaves, add the leaves to the scene
    if(settings.hasLeaves) {
        CS123SceneMaterial leafMaterial;
        leafMaterial.clear();
        leafMaterial.cAmbient.r = 0.05f;
        leafMaterial.cAmbient.g = 0.5f;
        leafMaterial.cAmbient.b = 0.05f;
        leafMaterial.cDiffuse.r = 0.05f;
        leafMaterial.cDiffuse.g = 0.3f;
        leafMaterial.cDiffuse.b = 0.2f;

        leafMaterial.cSpecular.r = leafMaterial.cSpecular.g = leafMaterial.cSpecular.b = 0.3;
        leafMaterial.shininess = 12;

        CS123ScenePrimitive leafCyl = {PrimitiveType::PRIMITIVE_CYLINDER, std::string(), leafMaterial};
        int numLeaves = m_lSystemViz->getNumLeaves();
        for(int i = 0; i < numLeaves; i++) {
            addPrimitive(leafCyl, m_lSystemViz->getLeafMatrix(i));
        }

    }
}
