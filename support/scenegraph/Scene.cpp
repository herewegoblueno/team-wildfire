#include "Scene.h"
#include "support/camera/Camera.h"
#include "support/lib/CS123ISceneParser.h"

#include "glm/gtx/transform.hpp"


Scene::Scene()
{
    primitiveCount = -1;
}

Scene::Scene(Scene &scene)
{
    setGlobal(scene.globalData);
    lightingInformation = scene.lightingInformation;
    primitives = scene.primitives;
    primitiveCount = scene.primitiveCount;
}

Scene::~Scene()
{
    // Do not delete m_camera, it is owned by SupportCanvas3D
}

void Scene::parse(Scene *sceneToFill, CS123ISceneParser *parser) {
   CS123SceneGlobalData global = { 1, 1, 1, 1};
   parser->getGlobalData(global);
   sceneToFill->setGlobal(global);

   CS123SceneLightData light;
   for (int lightID = 0; lightID < parser->getNumLights(); lightID++){
       parser->getLightData(lightID, light);
       sceneToFill->addLight(light);
   }

   traverseSceneGraph(sceneToFill, glm::mat4(1.f), parser->getRootNode());
    sceneToFill->primitiveCount = sceneToFill->primitives.size();
}

void Scene::addPrimitive(const CS123ScenePrimitive &scenePrimitive, const glm::mat4x4 &matrix) {
    primitives.push_back({scenePrimitive, matrix});
}

void Scene::addLight(const CS123SceneLightData &sceneLight) {
    lightingInformation.push_back(sceneLight);
}

void Scene::setGlobal(const CS123SceneGlobalData &global) {
    globalData = global;
}

void Scene::traverseSceneGraph (Scene *sceneToFill, glm::mat4 accumultedMatrix, CS123SceneNode *node){
    glm::mat4 localTransMat = glm::mat4(1.f);

    for (int i = node->transformations.size() - 1; i > -1; i--){
        CS123SceneTransformation* trans = node->transformations[i];
        switch(trans->type){
            case (TRANSFORMATION_SCALE):
                localTransMat = glm::scale(trans->scale) * localTransMat;
                break;
            case (TRANSFORMATION_ROTATE):
                localTransMat = glm::rotate(trans->angle, trans->rotate) * localTransMat;
                break;
            case (TRANSFORMATION_TRANSLATE):
                localTransMat = glm::translate(trans->translate) * localTransMat;
                break;
            case (TRANSFORMATION_MATRIX):
                localTransMat = trans->matrix * localTransMat;
                break;
        }
    }

    glm::mat4 accumulatedTransMat = accumultedMatrix * localTransMat;

    for (unsigned long i = 0; i < node->primitives.size(); i++){
        sceneToFill->addPrimitive(*node->primitives[i], accumulatedTransMat);
    }

    for (unsigned long i = 0; i < node->children.size(); i++){
        traverseSceneGraph(sceneToFill, accumulatedTransMat, node->children[i]);
    }
}


