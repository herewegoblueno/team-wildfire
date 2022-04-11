/**
 * @file    Settings.h
 *
 * This file contains various settings and enumerations that you will need to use in the various
 * assignments. The settings are bound to the GUI via static data bindings.
 */

#ifndef SETTINGS_H
#define SETTINGS_H

#include <QObject>
#include "support/lib/RGBA.h"


// Enumeration values for the Shapes that the user can select in the GUI.
enum ShapeType {
    SHAPE_CUBE,
    SHAPE_CONE,
    SHAPE_SPHERE,
    SHAPE_CYLINDER,
    SHAPE_TORUS,
    SHAPE_SPECIAL_1,
    SHAPE_SPECIAL_2,
    NUM_SHAPE_TYPES
};

enum UITab {
    SHADER_TESTING_TAB,
    SHADER_IMPORT_TAB,
    TREE_TESTING_TAB,
    GALLERY_TAB
};

// Enumeration values for the currently selected scene type
enum SceneMode {
    SCENEMODE_TREE_TESTING,
    SCENEMODE_SHADER_TESTING,
    SCENEMODE_SHADER_IMPORT,
    SCENEMODE_COMBINED_SCENE
};

// Enumeration values for the currently selected camera type
enum CameraMode {
    CAMERAMODE_ORBIT,
    CAMERAMODE_CAMTRANS
};

// You can access all app settings through the "settings" global variable.
struct Settings
{

    // Loads settings from disk, or fills in default values if no saved settings exist.
    void loadSettingsOrDefaults();

    // Saves the current settings to disk.
    void saveSettings();

    // Shapes
    //int shapeType;              // Selected shape type
    int shapeParameter1;
    int shapeParameter2;
    float shapeParameter3;
    bool useLighting;           // Enable default lighting
    bool drawWireframe;         // Draw wireframe only
    bool drawNormals;           // Turn normals on and off

    // Camtrans
    bool useOrbitCamera;        // Use the built-in orbiting camera instead of the Camtrans camera
    float cameraPosX;
    float cameraPosY;
    float cameraPosZ;
    float cameraRotU;
    float cameraRotV;
    float cameraRotN;
    float cameraFov;            // The camera's field of view, which is twice the height angle.
    float cameraNear;           // The distance from the camera to the near clipping plane.
    float cameraFar;            // The distance from the camera to the far clipping plane.

    // Modeler
    int objTool;                // The currently selected modeler tool.

    bool usePointLights;        // Enable or disable point lighting.
    bool useDirectionalLights;  // Enable or disable directional lighting (extra credit).
    bool useSpotLights;         // Enable or disable spot lights (extra credit).

    bool lengthStochasticity;
    bool angleStochasticity;
    bool hasLeaves;

    int numRecursions;
    int lSystemType;

    int getSceneMode();
    int getCameraMode();

    int currentTab;
};

// The global Settings object, will be initialized by MainWindow
extern Settings settings;

#endif // SETTINGS_H
