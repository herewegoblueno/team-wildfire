#ifndef SCENE_H
#define SCENE_H

#include "support/lib/CS123SceneData.h"
#include <vector>

class Camera;
class CS123ISceneParser;


/**
 * @class Scene
 *
 * @brief This is the base class for all scenes. Modify this class if you want to provide
 * common functionality to all your scenes.
 */
class Scene {
public:
    Scene();
    Scene(Scene &scene);
    virtual ~Scene();

    std::vector<CS123SceneLightData> lightingInformation;
    std::vector<CS123ScenePrimitiveBundle> primitives;
    CS123SceneGlobalData globalData;

    virtual void settingsChanged() {}

    static void parse(Scene *sceneToFill, CS123ISceneParser *parser);

protected:

    // Adds a primitive to the scene.
    virtual void addPrimitive(const CS123ScenePrimitive &scenePrimitive, const glm::mat4x4 &matrix);

    // Adds a light to the scene.
    virtual void addLight(const CS123SceneLightData &sceneLight);

    // Sets the global data for the scene.
    virtual void setGlobal(const CS123SceneGlobalData &global);

    static void traverseSceneGraph (Scene *sceneToFill, glm::mat4 accumultedMatrix, CS123SceneNode *node);

    int primitiveCount;
};

#endif // SCENE_H
