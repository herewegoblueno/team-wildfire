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

// Enumeration values for the currently selected UI tab
enum UITab {
    FOREST_TAB,
    FIRE_TAB
};

// Enumeration values for the currently selected scene type
enum SceneMode {
    FOREST_SCENE_MODE,
    FIRE_SCENE_MODE,
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

    // Forest
    int recursionDepth;
    float leafDensity;
    float branchStochasticity;

    //Forest vizualization
    bool visualizeForestVoxelGrid;
    float visualizeForestVoxelGridEyeX;
    float visualizeForestVoxelGridEyeY;
    float visualizeForestVoxelGridEyeZ;
    float visualizeForestVoxelGridEyeRadius;
    float visualizeForestVoxelGridMinTemp;
    float visualizeForestVoxelGridMaxTemp;

    // Shapes
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

    bool usePointLights;        // Enable or disable point lighting.
    bool useDirectionalLights;  // Enable or disable directional lighting (extra credit).
    bool useSpotLights;         // Enable or disable spot lights (extra credit).

    int getSceneMode();
    int getCameraMode();

    int currentTab;
};

// The global Settings object, will be initialized by MainWindow
extern Settings settings;

#endif // SETTINGS_H
