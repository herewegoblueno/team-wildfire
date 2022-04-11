#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H

#include "Scene.h"

// Maximum number of lights, as defined in shader.
const int MAX_NUM_LIGHTS = 10;

class SupportCanvas3D;

using std::string;

/**
 * @class  OpenGLScene
 *
 * Basic Scene abstract class that supports OpenGL. Students will extend this class in ShapesScene
 * and SceneviewScene.
 */
class OpenGLScene : public Scene {
public:
    virtual ~OpenGLScene();
    virtual void settingsChanged() override;
    virtual void render(SupportCanvas3D *context) = 0;

protected:

    void setClearColor();
    void setClearColor(float r, float g, float b, float a);
};

#endif // OPENGLSCENE_H
