/*!

 Settings.h
 CS123 Support Code

 @author  Evan Wallace (edwallac)
 @date    9/1/2010

 This file contains various settings and enumerations that you will need to
 use in the various assignments. The settings are bound to the GUI via static
 data bindings.

**/

#include "Settings.h"
#include <QFile>
#include <QSettings>

Settings settings;
std::string SETTINGS_ORGO = "TeamWildfire";
std::string SETTINGS_NAME = "WildfireSim";



/**
 * Loads the application settings, or, if no saved settings are available, loads default values for
 * the settings. You can change the defaults here.
 */
void Settings::loadSettingsOrDefaults() {
    // Set the default values below
    QSettings s(SETTINGS_ORGO.c_str(), SETTINGS_NAME.c_str());

    // Forest
    leafDensity = s.value("leafDensity", 1.0).toDouble();
    branchStochasticity = s.value("branchStochasticity", 0.5).toDouble();

    //Forest Visualization (some of these are not saved)
    visualizeForestVoxelGrid = s.value("visualizeForestVoxelGrid", false).toBool();
    voxelGridMode = static_cast<VoxelVisualizationModes>(s.value("voxelGridMode", TEMPERATURE).toInt());

    visualizeVectorField = s.value("visualizeVectorField", false).toBool();
    vectorGridMode = static_cast<VectorFieldVisualizationModes>(s.value("vectorGridMode", UFIELD).toInt());

    seeBranchModules = s.value("seeBranchModules", false).toBool();
    hideSelectedModuleHighlight = s.value("hideSelectedModuleHighlight", false).toBool();
    moduleVisualizationMode = static_cast<ModuleVisualizationModes>(s.value("moduleVisualizationMode", ID).toInt());

    visualizeForestVoxelGridEyeX =  s.value("visualizeForestVoxelGridEyeX", 0).toDouble();
    visualizeForestVoxelGridEyeY =  s.value("visualizeForestVoxelGridEyeY", 0).toDouble();
    visualizeForestVoxelGridEyeZ =  s.value("visualizeForestVoxelGridEyeZ", 0).toDouble();
    visualizeForestVoxelGridEyeRadius =  s.value("visualizeForestVoxelGridEyeRadius", 0.5).toDouble();
    visualizeForestVoxelGridMinTemp = s.value("visualizeForestVoxelGridMinTemp", 0).toDouble();
    visualizeForestVoxelGridMaxTemp = s.value("visualizeForestVoxelGridMaxTemp", 3).toDouble();

    //It's actually important not to save these two, since a valid module id depends on the currently available IDs, which changes
    //With every run
    visualizeOnlyVoxelsTouchingSelectedModule =  s.value("visualizeOnlyVoxelsTouchingSelectedModule", false).toBool();
    selectedModuleId = s.value("selectedModuleId", DEFAULT_MODULE_ID).toInt();

    simulatorTimescale = s.value("simulatorTimescale", 1).toDouble();

    // Shape Tesselation Settings
    shapeParameter1 = s.value("shapeParameter1", 15).toInt();
    shapeParameter2 = s.value("shapeParameter2", 15).toInt();
    shapeParameter3 = s.value("shapeParameter3", 15).toDouble();

    //Rendering Settings
    drawWireframe = s.value("drawWireframe", false).toBool();
    drawNormals = s.value("drawNormals", false).toBool();
    useLighting = s.value("useLighting", true).toBool();
    usePointLights = s.value("usePointLights", true).toBool();

    // Camera Settings
    useOrbitCamera = s.value("useOrbitCamera", true).toBool();
    cameraFov = s.value("cameraFov", 55).toDouble();
    cameraNear = s.value("cameraNear", 0.1).toDouble();
    cameraFar = s.value("cameraFar", 50).toDouble();

    //UI serrings
    currentTab = s.value("currentTab", FOREST_TAB).toInt();

    // These are for computing deltas and the values don't matter, so start all dials in the up
    // position
    cameraPosX = 0;
    cameraPosY = 0;
    cameraPosZ = 0;
    cameraRotU = 0;
    cameraRotV = 0;
    cameraRotN = 0;
}

void Settings::saveSettings() {
    QSettings s(SETTINGS_ORGO.c_str(), SETTINGS_NAME.c_str());

    // Forest 
    s.setValue("visualizeForestVoxelGrid", visualizeForestVoxelGrid);
    s.setValue("visualizeVectorField", visualizeVectorField);
    s.setValue("voxelGridMode", voxelGridMode);
    s.setValue("vectorGridMode", vectorGridMode);
    s.setValue("moduleVisualizationMode", moduleVisualizationMode);

    s.setValue("simulatorTimescale", simulatorTimescale);
    s.setValue("visualizeForestVoxelGridMinTemp", visualizeForestVoxelGridMinTemp);
    s.setValue("visualizeForestVoxelGridMaxTemp", visualizeForestVoxelGridMaxTemp);


    // Shapes
    s.setValue("shapeParameter1", shapeParameter1);
    s.setValue("shapeParameter2", shapeParameter2);
    s.setValue("shapeParameter3", shapeParameter3);
    s.setValue("useLighting", useLighting);
    s.setValue("drawWireframe", drawWireframe);
    s.setValue("drawNormals", drawNormals);

    // Camtrans
    s.setValue("useOrbitCamera", useOrbitCamera);
    s.setValue("cameraFov", cameraFov);
    s.setValue("cameraNear", cameraNear);
    s.setValue("cameraFar", cameraFar);

    s.setValue("usePointLights", usePointLights);
    s.setValue("useDirectionalLights", useDirectionalLights);
    s.setValue("useSpotLights", useSpotLights);

    s.setValue("currentTab", currentTab);
}

int Settings::getSceneMode() {
    if (this->currentTab == FOREST_TAB) {
        return FOREST_SCENE_MODE;
    }
    if (this->currentTab == FIRE_TAB) {
        return FIRE_SCENE_MODE;
    }
    return -1; //Should never happen
}

int Settings::getCameraMode() {
    if (this->useOrbitCamera)
        return CAMERAMODE_ORBIT;
    else
        return CAMERAMODE_CAMTRANS;
}
