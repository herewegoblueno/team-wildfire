#include "GalleryScene.h"

#include "GL/glew.h"
#include <QGLWidget>
#include "support/camera/Camera.h"

#include "support/Settings.h"
#include "support/scenegraph/SupportCanvas3D.h"
#include "support/lib/ResourceLoader.h"
#include "support/gl/shaders/CS123Shader.h"
#include "ShaderEvolutionTestingScene.h"

#include "support/shapes/Cube.h"
#include "support/shapes/Cone.h"
#include "support/shapes/Sphere.h"
#include "support/shapes/Cylinder.h"
#include "support/shapes/Torus.h"

#include "glm/gtx/transform.hpp"
#include "time.h"


#include <iostream>
using namespace CS123::GL;


GalleryScene::GalleryScene()
{
    treeTypeDist = std::uniform_int_distribution<>(0, 5);
    levelDist = std::uniform_int_distribution<>(2, 4);
    RNG.seed(time(NULL));

    loadPhongShader();
    numTreePrims = 0;

    shapeBank.resize(6);
    defineShapeBank();
    setUpLights();
    makePotPosns();

    loadScene();
}


GalleryScene::~GalleryScene()
{
}

void GalleryScene::settingsChanged() {
    // make a new LSystem with the current settings
    loadScene();
}

void::GalleryScene::loadPhongShader() {
    std::string vertexSource = ResourceLoader::loadResourceFileToString(":/shaders/default.vert");
    std::string fragmentSource = ResourceLoader::loadResourceFileToString(":/shaders/default.frag");
    m_phongShader = std::make_unique<CS123Shader>(vertexSource, fragmentSource);
}


void GalleryScene::setUpLights() {
    // add some lights to the scene
    // to do: make the lights in a helper
    CS123SceneLightData mainLight = {0, LightType::LIGHT_POINT, glm::vec4(0.05), glm::vec3(0, 0, 10), glm::vec4(0, 5, 0, 0)};

    CS123SceneLightData light = {1, LightType::LIGHT_POINT, glm::vec4(0.5, 0.4, 0, 1), glm::vec3(1, 0, 1), glm::vec4(5, 3, 5, 0)};
    CS123SceneLightData light2 = {2, LightType::LIGHT_POINT, glm::vec4(0.3, 0.3, 1, 1), glm::vec3(1, 0, 1), glm::vec4(-5, 3, 5, 0)};
    CS123SceneLightData light3 = {3, LightType::LIGHT_POINT, glm::vec4(1, 0.3, 0.3, 1), glm::vec3(1, 0, 1), glm::vec4(0, 3, -5, 0)};


    lightingInformation.push_back(light);
    lightingInformation.push_back(light2);
    lightingInformation.push_back(light3);
    lightingInformation.push_back(mainLight);

    CS123SceneGlobalData globalLSys = {0.55f, 0.8, 0.8, 1};
    setGlobal(globalLSys);
}


void GalleryScene::render(SupportCanvas3D *context) {
    setClearColor(153.f/255, 229.f/255, 255.f/255, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::vector<int> indexesForSpecialShaders = {0, 2, 4, 6, 8};

    //Render the pots with the custom shaders
    renderNonPhongGeometry(context, indexesForSpecialShaders);


    //Render everything else witht eh phong shader
    m_phongShader->bind();
    setPhongSceneUniforms(context);
    setLights();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    renderPhongGeometry(indexesForSpecialShaders);
    glBindTexture(GL_TEXTURE_2D, 0);
    m_phongShader->unbind();

    //Trigger another render
    context->update();
}


void GalleryScene::renderNonPhongGeometry(SupportCanvas3D *context, std::vector<int> indexes){
    std::vector<std::unique_ptr<CS123::GL::CS123Shader>> * programs =
            context->getShaderScene()->getShaderPrograms();
    for (int i = 0; i < indexes.size(); i ++){

        //Getting the shape
        CS123ScenePrimitiveBundle bundle = primitives[indexes[i]];
        CS123SceneMaterial mat = bundle.primitive.material;
        mat.cDiffuse *= globalData.kd;
        mat.cAmbient *= globalData.ka;
        mat.shininess *= globalData.ks;
        mat.cTransparent *= globalData.kt;


        CS123Shader *current_shader = programs->at(i).get();
        current_shader->bind();

        //Setting uniforms
        Camera *camera = context->getCamera();
        current_shader->setUniform("useLighting", settings.useLighting);
        current_shader->setUniform("p" , camera->getProjectionMatrix());
        current_shader->setUniform("v", camera->getViewMatrix());

        //Setting lights
        int size = lightingInformation.size();
        for (int i = 0; i < size; i++){
            current_shader->setLight(lightingInformation[i]);
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        current_shader->setUniform("m", bundle.model);
        current_shader->setUniform("time", ShaderEvolutionTestingScene::calculateTime());
        current_shader->applyMaterial(mat);
        (shapeBank[(int) bundle.primitive.type])->draw();
        glBindTexture(GL_TEXTURE_2D, 0);
        current_shader->unbind();
    }
}

void GalleryScene::setPhongSceneUniforms(SupportCanvas3D *context) {
    Camera *camera = context->getCamera();
    m_phongShader->setUniform("useLighting", settings.useLighting);
    m_phongShader->setUniform("p" , camera->getProjectionMatrix());
    m_phongShader->setUniform("v", camera->getViewMatrix());
}

void GalleryScene::setLights()
{
    int size = lightingInformation.size();
    for (int i = 0; i < size; i++){
        m_phongShader->setLight(lightingInformation[i]);
    }
}

// this could probably be a constant but the syntax is slightly less ugly this way
void::GalleryScene::makePotPosns() {
    m_potPosns.push_back(glm::vec3(2, -1.085, 2));
    m_potPosns.push_back(glm::vec3(1.5f, -1.085, 0));
    m_potPosns.push_back(glm::vec3(0, -1.085, -2));
    m_potPosns.push_back(glm::vec3(-1.5f, -1.085, 0));
    m_potPosns.push_back(glm::vec3(-2, -1.085, 2));
}


void GalleryScene::loadScene() {
    primitives.clear();
    numTreePrims = 0;

    int oldRecurs = settings.numRecursions;
    int oldType = settings.lSystemType;
    bool oldLeaves = settings.hasLeaves;
    int numTrees = m_potPosns.size();

    //Furniture added first, trees added last
    //Code assumes that the first 5 primitices are the pots so that it can add the
    //evolved shaders to them
    addPotsToScene();
    addGroundToScene();
    //addBackgroundToScene();

    // make new L System visualizers
    for(int i = 0; i < numTrees; i++) {
        // get recursive depth between 2 and 4
        settings.numRecursions = levelDist(RNG);
        // randomize type of tree (0-5)
        settings.lSystemType = treeTypeDist(RNG);
        // if seaweed, cap depth at 3 bc it ar large recursion levels
        if(settings.lSystemType == 1 && settings.numRecursions > 3) {
            settings.numRecursions = 3;
        }

        m_lSystemViz = std::make_unique<LSystemVisualizer>();
        makeLSystemVisualizer(i);
    }

    settings.numRecursions = oldRecurs;
    settings.lSystemType = oldType;
    settings.hasLeaves = oldLeaves;
}

void GalleryScene::makeLSystemVisualizer(int index) {

    glm::mat4x4 outerTrans = glm::translate(glm::vec3(m_potPosns.at(index).x, 0, m_potPosns.at(index).z));

    //Bettr rotation for the trees to face us
    outerTrans = outerTrans * glm::rotate((float)M_PI/4.f, glm::vec3(0, 1, 0));

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

        // also seaweed can't have leaves
        settings.hasLeaves = false;
    } else {
        settings.hasLeaves = true;
    }

    // add all cylinders to scene
    CS123ScenePrimitive cyl = {PrimitiveType::PRIMITIVE_CYLINDER, "", material};
    for(int i = 0; i < numCyls; i++) {
        // make a new scene primitive
        numTreePrims ++;
        addPrimitive(cyl, outerTrans * m_lSystemViz->getTransformationMatrix(i));

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

        CS123ScenePrimitive leafCyl = {PrimitiveType::PRIMITIVE_CYLINDER, "", leafMaterial};
        int numLeaves = m_lSystemViz->getNumLeaves();
        for(int i = 0; i < numLeaves; i++) {
            numTreePrims ++;
            addPrimitive(leafCyl, outerTrans*m_lSystemViz->getLeafMatrix(i));
        }

    }

    settings.hasLeaves = true;
}


// adds all pots to scene
void::GalleryScene::addPotsToScene() {
    int numPots = m_potPosns.size();

    CS123SceneMaterial material;
    material.clear();
    material.cAmbient.g = 0.1f;
    material.cAmbient.b = 0.05f;
    material.cDiffuse.g = 0.7f;
    material.cDiffuse.b = 0.5f;

    CS123ScenePrimitive pot = {PrimitiveType::PRIMITIVE_CYLINDER, "", material};
    glm::mat4x4 potTransformation = glm::scale(glm::vec3(0.8f, 1.f, 0.9f));

    CS123ScenePrimitive rim = {PrimitiveType::PRIMITIVE_TORUS, "", material};
    glm::mat4x4 rimTransformation =
            glm::translate(glm::vec3(0, -0.45, 0))
            * glm::scale(glm::vec3(0.04))
            * glm::rotate((float)M_PI_2, glm::vec3(1, 0, 0));


    for(int i = 0; i < numPots; i++) {
        addPrimitive(pot, glm::translate(m_potPosns.at(i)) * potTransformation);
        addPrimitive(rim, glm::translate(m_potPosns.at(i)) * rimTransformation);
    }
}

void GalleryScene::addGroundToScene() {
    // make a large, flat cube
    glm::vec3 pos = glm::vec3(0, -1.5f, 0);
    glm::vec3 scale = glm::vec3(30, 0.04f, 30);

    CS123SceneMaterial material;
    material.clear();
    material.cAmbient.r = 0.45;
    material.cAmbient.g = .45;
    material.cAmbient.b = .45;
    material.cDiffuse.r = 0.4f;
    material.cDiffuse.g = 0.4f;
    material.cDiffuse.b = 0.4f;

    CS123ScenePrimitive theGround = {PrimitiveType::PRIMITIVE_CUBE, "", material};
    addPrimitive(theGround, glm::translate(pos) * glm::scale(scale));
}

//void GalleryScene::addBackgroundToScene() {
//    // make large, flat cubes off in the distance
//    glm::vec3 scale = glm::vec3(30, 30, 0.04f);

//    // make it sky colored
//    CS123SceneMaterial material;
//    material.clear();
//    material.cAmbient.r = 0.2f;
//    material.cAmbient.g = 0.35f;
//    material.cAmbient.b = 0.45f;
//    material.cDiffuse.r = 0.55f;
//    material.cDiffuse.g = 0.6f;
//    material.cDiffuse.b = 1.f;

//    CS123ScenePrimitive theSky = {PrimitiveType::PRIMITIVE_CUBE, "", material};
//    addPrimitive(theSky, glm::translate(glm::vec3(0, 0, -15)) * glm::scale(scale));
//    addPrimitive(theSky, glm::translate(glm::vec3(0, 0, 15)) * glm::scale(scale));
//    addPrimitive(theSky, glm::translate(glm::vec3(15, 0, 0)) * glm::rotate((float)M_PI_2, glm::vec3(0, 1, 0)) *  glm::scale(scale));
//    addPrimitive(theSky, glm::translate(glm::vec3(-15, 0, 0)) * glm::rotate((float)M_PI_2, glm::vec3(0, 1, 0)) *  glm::scale(scale));
//}

void GalleryScene::renderPhongGeometry(std::vector<int> indexesToSkip) {
    int size = primitives.size();
    for (int i = 0; i < size; i++){
        if (std::find(indexesToSkip.begin(), indexesToSkip.end(), i) != indexesToSkip.end()) continue;
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

void GalleryScene::defineShapeBank(){
    //Can be linked to settings.parameter1-3, but since
    //we know the scenes that are being made we'll but hardcode it
    //Can be helpful for perfomance reasons too
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

