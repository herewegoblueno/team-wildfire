QT += opengl xml widgets
TARGET = wildfire
TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++14
CONFIG += c++14


unix:!macx {
    LIBS += -lGLU
}
macx {
    QMAKE_CFLAGS_X86_64 += -mmacosx-version-min=10.7
    QMAKE_CXXFLAGS_X86_64 = $$QMAKE_CFLAGS_X86_64
    CONFIG += c++11
    ICON = $$PWD/killWithFire.icns
}
win32 {
    DEFINES += GLEW_STATIC
    LIBS += -lopengl32 -lglu32
    INCLUDEPATH +=  ../eigen-master
}

SOURCES += ui/mainwindow.cpp \
    fire/firemanager.cpp \
    fire/smoke.cpp \
    main.cpp \
    glew-1.10.0/src/glew.c \
    simulation/physics.cpp \
    simulation/simulationheattransfer.cpp \
    simulation/simulationwaterphysics.cpp \
    simulation/simulationwindfield.cpp \
    simulation/simulator.cpp \
    support/Settings.cpp \
    support/camera/CamtransCamera.cpp \
    support/camera/OrbitingCamera.cpp \
    support/camera/QuaternionCamera.cpp \
    support/gl/GLDebug.cpp \
    support/gl/datatype/FBO.cpp \
    support/gl/datatype/IBO.cpp \
    support/gl/datatype/VAO.cpp \
    support/gl/datatype/VBO.cpp \
    support/gl/datatype/VBOAttribMarker.cpp \
    support/gl/shaders/Shader.cpp \
    support/gl/shaders/CS123Shader.cpp \
    support/gl/textures/DepthBuffer.cpp \
    support/gl/textures/RenderBuffer.cpp \
    support/gl/textures/Texture.cpp \
    support/gl/textures/Texture2D.cpp \
    support/gl/textures/TextureParameters.cpp \
    support/gl/textures/TextureParametersBuilder.cpp \
    support/gl/util/FullScreenQuad.cpp \
    support/lib/CS123XmlSceneParser.cpp \
    support/lib/RGBA.cpp \
    support/lib/ResourceLoader.cpp \
    support/scenegraph/BasicFireScene.cpp \
    support/scenegraph/BasicForestScene.cpp \
    support/scenegraph/OpenGLScene.cpp \
    support/scenegraph/Scene.cpp \
    support/scenegraph/SupportCanvas3D.cpp \
    support/shapes/CircleBase.cpp \
    support/shapes/Cone.cpp \
    support/shapes/Cube.cpp \
    support/shapes/Cylinder.cpp \
    support/shapes/Fruit.cpp \
    support/shapes/Leaf.cpp \
    support/shapes/OpenGLShape.cpp \
    support/shapes/Shape.cpp \
    support/shapes/Sphere.cpp \
    support/shapes/Tessellator.cpp \
    support/shapes/Trunk.cpp \
    trees/LSystem.cpp \
    trees/TreeGenerator.cpp \
    trees/forest.cpp \
    trees/module.cpp \
    trees/terrain.cpp \
    fire/fire.cpp \
    ui/extrawidgets/ctkrangeslider.cpp \
    voxels/voxel.cpp \
    voxels/voxelgrid.cpp \
    voxels/voxelgridline.cpp


HEADERS += ui/mainwindow.h \
    fire/firemanager.h \
    fire/particle.h \
    fire/smoke.h \
    simulation/fluid.h \
    simulation/physics.h \
    simulation/simulator.h \
    support/Settings.h \
    support/camera/Camera.h \
    support/camera/CamtransCamera.h \
    support/camera/OrbitingCamera.h \
    support/camera/QuaternionCamera.h \
    support/gl/GLDebug.h \
    support/gl/datatype/FBO.h \
    support/gl/datatype/IBO.h \
    support/gl/datatype/VAO.h \
    support/gl/datatype/VBO.h \
    support/gl/datatype/VBOAttribMarker.h \
    support/gl/shaders/Shader.h \
    support/gl/shaders/CS123Shader.h \
    support/gl/shaders/ShaderAttribLocations.h \
    support/gl/textures/DepthBuffer.h \
    support/gl/textures/RenderBuffer.h \
    support/gl/textures/Texture.h \
    support/gl/textures/Texture2D.h \
    support/gl/textures/TextureParameters.h \
    support/gl/textures/TextureParametersBuilder.h \
    support/gl/util/FullScreenQuad.h \
    support/lib/CS123ISceneParser.h \
    support/lib/CS123SceneData.h \
    support/lib/CS123XmlSceneParser.h \
    support/lib/RGBA.h \
    support/lib/ResourceLoader.h \
    support/scenegraph/BasicFireScene.h \
    support/scenegraph/BasicForestScene.h \
    support/scenegraph/OpenGLScene.h \
    support/scenegraph/Scene.h \
    support/scenegraph/SupportCanvas3D.h \
    support/shapes/CircleBase.h \
    support/shapes/Cone.h \
    support/shapes/Cube.h \
    support/shapes/Cylinder.h \
    support/shapes/Fruit.h \
    support/shapes/Leaf.h \
    support/shapes/OpenGLShape.h \
    support/shapes/Shape.h \
    support/shapes/Sphere.h \
    support/shapes/Tessellator.h \
    support/shapes/TriMesh.h \
    support/shapes/Trunk.h \
    trees/LSystem.h \
    trees/TreeGenerator.h \
    utils/Random.h \
    trees/forest.h \
    trees/module.h \
    trees/terrain.h \
    ui/extrawidgets/ctkrangeslider.h \
    ui_mainwindow.h \
    glew-1.10.0/include/GL/glew.h \
    fire/fire.h \
    voxels/voxel.h \
    voxels/voxelgrid.h \
    voxels/voxelgridline.h

FORMS += ui/mainwindow.ui
INCLUDEPATH += glm ui glew-1.10.0/include
DEPENDPATH += glm ui glew-1.10.0/include

DEFINES += _USE_MATH_DEFINES
DEFINES += TIXML_USE_STL
DEFINES += GLM_SWIZZLE GLM_FORCE_RADIANS

# Don't add the -pg flag unless you know what you are doing. It makes QThreadPool freeze on Mac OS X
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS_WARN_ON -= -Wall
QMAKE_CXXFLAGS_WARN_ON += -Waddress -Warray-bounds -Wc++0x-compat -Wchar-subscripts -Wformat\
                          -Wmain -Wmissing-braces -Wparentheses -Wreorder -Wreturn-type \
                          -Wsequence-point -Wsign-compare -Wstrict-overflow=1 -Wswitch \
                          -Wtrigraphs -Wuninitialized -Wunused-label -Wunused-variable \
                          -Wvolatile-register-var -Wno-extra
win32 {
QMAKE_CXXFLAGS_WARN_ON -= -Waddress -Warray-bounds -Wc++0x-compat -Wchar-subscripts -Wformat\
                          -Wmain -Wmissing-braces -Wparentheses -Wreorder -Wreturn-type \
                          -Wsequence-point -Wsign-compare -Wstrict-overflow=1 -Wswitch \
                          -Wtrigraphs -Wuninitialized -Wunused-label -Wunused-variable \
                          -Wvolatile-register-var -Wno-extra
QMAKE_CXXFLAGS += -Wno-enum-compare
}



QMAKE_CXXFLAGS += -g

# QMAKE_CXX_FLAGS_WARN_ON += -Wunknown-pragmas -Wunused-function -Wmain

macx {
    QMAKE_CXXFLAGS_WARN_ON -= -Warray-bounds -Wc++0x-compat
}

RESOURCES += \
    resources.qrc

DISTFILES += \
    xmlScenes/basicScene.xml

# For Eigen, see readme for more details
macx {
    HOME_DIR = $$(HOME)
}

win32 {
    HOME_DIR = $$(HOMEPATH)
}

QMAKE_CXXFLAGS = -I "$${HOME_DIR}/eigen-git-mirror"
INCLUDEPATH += "$${HOME_DIR}/eigen-git-mirror"
DEPENDPATH += "$${HOME_DIR}/eigen-git-mirror"

include(cuda_lib/cuda_lib.pri)
