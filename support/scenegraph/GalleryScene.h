#ifndef GALLERYSCENE_H
#define GALLERYSCENE_H

#include "OpenGLScene.h"

#include <memory>
#include <vector>
#include "support/shapes/Shape.h"
#include "lsystems/LSystemVisualizer.h"
#include "LSystemTreeScene.h"
#include <random>

// basically the L System tree scene but copied
//and modified to accomodate for a whole scene

namespace CS123 { namespace GL {

    class Shader;
    class CS123Shader;
    class Texture2D;
}}



class GalleryScene : public OpenGLScene {
public:
    GalleryScene();
    virtual ~GalleryScene();
    virtual void render(SupportCanvas3D *context) override;
    virtual void settingsChanged() override;

    // Use this method to set an internal selection, based on the (x, y) position of the mouse
    // pointer.  This will be used during the "modeler" lab, so don't worry about it for now.
    void setSelection(int x, int y);
private:

    void loadPhongShader();

    void setPhongSceneUniforms(SupportCanvas3D *context);
    void setLights();
    void renderNonPhongGeometry(SupportCanvas3D *context, std::vector<int> indexesForSpecialShaders);
    void renderPhongGeometry(std::vector<int> indexesToSkip);
    void defineShapeBank();
    void makeLSystemVisualizer(int index);

    void makePotPosns();
    void setUpLights();
    void loadScene();
    void addGroundToScene();
    void addPotsToScene();
//    void addBackgroundToScene();

    std::vector<std::unique_ptr<Shape>> shapeBank;
    std::unique_ptr<CS123::GL::CS123Shader> m_phongShader;
    std::unique_ptr<LSystemVisualizer> m_lSystemViz;

    std::vector<glm::vec3> m_potPosns;

    // could be useful for determining number of shapes related to trees maybe
    // meh
    int numTreePrims;


    std::minstd_rand RNG;
    std::uniform_int_distribution<> treeTypeDist;
    std::uniform_int_distribution<> levelDist;
};



#endif // GALLERYSCENE_H
