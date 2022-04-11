# -------------------------------------------------
# Project created by QtCreator 2010-08-22T14:12:19
# -------------------------------------------------
QT += opengl xml \
    widgets
TARGET = beagle
TEMPLATE = app

QMAKE_CXXFLAGS += -std=c++14
CONFIG += c++14

unix:!macx {
    LIBS += -lGLU
}
macx {
    QMAKE_CFLAGS_X86_64 += -mmacosx-version-min=10.7
    QMAKE_CXXFLAGS_X86_64 = $$QMAKE_CFLAGS_X86_64
    ICON = beagle_icon.icns
    CONFIG += c++11
}
win32 {
    DEFINES += GLEW_STATIC
    LIBS += -lopengl32 -lglu32
}

SOURCES += ui/mainwindow.cpp \
    lsystems/LSystem.cpp \
    lsystems/LSystemVisualizer.cpp \
    lsystems/Turtle.cpp \
    main.cpp \
    glew-1.10.0/src/glew.c \
    shaderevolution/AstNodes.cpp \
    shaderevolution/MutationFactory.cpp \
    shaderevolution/NodeDispenser.cpp \
    shaderevolution/ShaderConstructor.cpp \
    shaderevolution/ShaderEvolutionManager.cpp \
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
    support/gl/shaders/CS123Shader.cpp \
    support/gl/shaders/Shader.cpp \
    support/gl/textures/DepthBuffer.cpp \
    support/gl/textures/RenderBuffer.cpp \
    support/gl/textures/Texture.cpp \
    support/gl/textures/Texture2D.cpp \
    support/gl/textures/TextureParameters.cpp \
    support/gl/textures/TextureParametersBuilder.cpp \
    support/gl/util/FullScreenQuad.cpp \
    support/glm/detail/dummy.cpp \
    support/glm/detail/glm.cpp \
    support/lib/CS123XmlSceneParser.cpp \
    support/lib/RGBA.cpp \
    support/lib/ResourceLoader.cpp \
    support/scenegraph/LSystemTreeScene.cpp \
    support/scenegraph/OpenGLScene.cpp \
    support/scenegraph/Scene.cpp \
    support/scenegraph/ShaderEvolutionTestingScene.cpp \
    support/scenegraph/ShaderImportScene.cpp \
    support/scenegraph/SupportCanvas3D.cpp \
    support/shapes/CircularPlane.cpp \
    support/shapes/Cone.cpp \
    support/shapes/Cube.cpp \
    support/shapes/Cylinder.cpp \
    support/shapes/HemispherePlane.cpp \
    support/shapes/Loop.cpp \
    support/shapes/Rectplane.cpp \
    support/shapes/Shape.cpp \
    support/shapes/Sphere.cpp \
    support/shapes/Surface.cpp \
    support/shapes/Torus.cpp \
    ui/shadercodedisplayer.cpp \
    lsystems/LSystemUtils.cpp \
    support/scenegraph/GalleryScene.cpp


HEADERS += ui/mainwindow.h \
    lsystems/LSystem.h \
    lsystems/LSystemVisualizer.h \
    lsystems/Turtle.h \
    shaderevolution/AstNodes.h \
    shaderevolution/MutationFactory.h \
    shaderevolution/NodeDispenser.h \
    shaderevolution/ShaderConstructor.h \
    shaderevolution/ShaderEvolutionManager.h \
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
    support/gl/shaders/CS123Shader.h \
    support/gl/shaders/Shader.h \
    support/gl/shaders/ShaderAttribLocations.h \
    support/gl/textures/DepthBuffer.h \
    support/gl/textures/RenderBuffer.h \
    support/gl/textures/Texture.h \
    support/gl/textures/Texture2D.h \
    support/gl/textures/TextureParameters.h \
    support/gl/textures/TextureParametersBuilder.h \
    support/gl/util/FullScreenQuad.h \
    support/glm/common.hpp \
    support/glm/detail/_features.hpp \
    support/glm/detail/_fixes.hpp \
    support/glm/detail/_literals.hpp \
    support/glm/detail/_noise.hpp \
    support/glm/detail/_swizzle.hpp \
    support/glm/detail/_swizzle_func.hpp \
    support/glm/detail/_vectorize.hpp \
    support/glm/detail/func_common.hpp \
    support/glm/detail/func_common.inl \
    support/glm/detail/func_exponential.hpp \
    support/glm/detail/func_exponential.inl \
    support/glm/detail/func_geometric.hpp \
    support/glm/detail/func_geometric.inl \
    support/glm/detail/func_integer.hpp \
    support/glm/detail/func_integer.inl \
    support/glm/detail/func_matrix.hpp \
    support/glm/detail/func_matrix.inl \
    support/glm/detail/func_noise.hpp \
    support/glm/detail/func_noise.inl \
    support/glm/detail/func_packing.hpp \
    support/glm/detail/func_packing.inl \
    support/glm/detail/func_trigonometric.hpp \
    support/glm/detail/func_trigonometric.inl \
    support/glm/detail/func_vector_relational.hpp \
    support/glm/detail/func_vector_relational.inl \
    support/glm/detail/hint.hpp \
    support/glm/detail/intrinsic_common.hpp \
    support/glm/detail/intrinsic_common.inl \
    support/glm/detail/intrinsic_exponential.hpp \
    support/glm/detail/intrinsic_exponential.inl \
    support/glm/detail/intrinsic_geometric.hpp \
    support/glm/detail/intrinsic_geometric.inl \
    support/glm/detail/intrinsic_integer.hpp \
    support/glm/detail/intrinsic_integer.inl \
    support/glm/detail/intrinsic_matrix.hpp \
    support/glm/detail/intrinsic_matrix.inl \
    support/glm/detail/intrinsic_trigonometric.hpp \
    support/glm/detail/intrinsic_trigonometric.inl \
    support/glm/detail/intrinsic_vector_relational.hpp \
    support/glm/detail/intrinsic_vector_relational.inl \
    support/glm/detail/precision.hpp \
    support/glm/detail/precision.inl \
    support/glm/detail/setup.hpp \
    support/glm/detail/type_float.hpp \
    support/glm/detail/type_gentype.hpp \
    support/glm/detail/type_gentype.inl \
    support/glm/detail/type_half.hpp \
    support/glm/detail/type_half.inl \
    support/glm/detail/type_int.hpp \
    support/glm/detail/type_mat.hpp \
    support/glm/detail/type_mat.inl \
    support/glm/detail/type_mat2x2.hpp \
    support/glm/detail/type_mat2x2.inl \
    support/glm/detail/type_mat2x3.hpp \
    support/glm/detail/type_mat2x3.inl \
    support/glm/detail/type_mat2x4.hpp \
    support/glm/detail/type_mat2x4.inl \
    support/glm/detail/type_mat3x2.hpp \
    support/glm/detail/type_mat3x2.inl \
    support/glm/detail/type_mat3x3.hpp \
    support/glm/detail/type_mat3x3.inl \
    support/glm/detail/type_mat3x4.hpp \
    support/glm/detail/type_mat3x4.inl \
    support/glm/detail/type_mat4x2.hpp \
    support/glm/detail/type_mat4x2.inl \
    support/glm/detail/type_mat4x3.hpp \
    support/glm/detail/type_mat4x3.inl \
    support/glm/detail/type_mat4x4.hpp \
    support/glm/detail/type_mat4x4.inl \
    support/glm/detail/type_vec.hpp \
    support/glm/detail/type_vec.inl \
    support/glm/detail/type_vec1.hpp \
    support/glm/detail/type_vec1.inl \
    support/glm/detail/type_vec2.hpp \
    support/glm/detail/type_vec2.inl \
    support/glm/detail/type_vec3.hpp \
    support/glm/detail/type_vec3.inl \
    support/glm/detail/type_vec4.hpp \
    support/glm/detail/type_vec4.inl \
    support/glm/exponential.hpp \
    support/glm/ext.hpp \
    support/glm/fwd.hpp \
    support/glm/geometric.hpp \
    support/glm/glm.hpp \
    support/glm/gtc/constants.hpp \
    support/glm/gtc/constants.inl \
    support/glm/gtc/epsilon.hpp \
    support/glm/gtc/epsilon.inl \
    support/glm/gtc/matrix_access.hpp \
    support/glm/gtc/matrix_access.inl \
    support/glm/gtc/matrix_integer.hpp \
    support/glm/gtc/matrix_inverse.hpp \
    support/glm/gtc/matrix_inverse.inl \
    support/glm/gtc/matrix_transform.hpp \
    support/glm/gtc/matrix_transform.inl \
    support/glm/gtc/noise.hpp \
    support/glm/gtc/noise.inl \
    support/glm/gtc/packing.hpp \
    support/glm/gtc/packing.inl \
    support/glm/gtc/quaternion.hpp \
    support/glm/gtc/quaternion.inl \
    support/glm/gtc/random.hpp \
    support/glm/gtc/random.inl \
    support/glm/gtc/reciprocal.hpp \
    support/glm/gtc/reciprocal.inl \
    support/glm/gtc/type_precision.hpp \
    support/glm/gtc/type_precision.inl \
    support/glm/gtc/type_ptr.hpp \
    support/glm/gtc/type_ptr.inl \
    support/glm/gtc/ulp.hpp \
    support/glm/gtc/ulp.inl \
    support/glm/gtx/associated_min_max.hpp \
    support/glm/gtx/associated_min_max.inl \
    support/glm/gtx/bit.hpp \
    support/glm/gtx/bit.inl \
    support/glm/gtx/closest_point.hpp \
    support/glm/gtx/closest_point.inl \
    support/glm/gtx/color_space.hpp \
    support/glm/gtx/color_space.inl \
    support/glm/gtx/color_space_YCoCg.hpp \
    support/glm/gtx/color_space_YCoCg.inl \
    support/glm/gtx/compatibility.hpp \
    support/glm/gtx/compatibility.inl \
    support/glm/gtx/component_wise.hpp \
    support/glm/gtx/component_wise.inl \
    support/glm/gtx/constants.hpp \
    support/glm/gtx/dual_quaternion.hpp \
    support/glm/gtx/dual_quaternion.inl \
    support/glm/gtx/epsilon.hpp \
    support/glm/gtx/euler_angles.hpp \
    support/glm/gtx/euler_angles.inl \
    support/glm/gtx/extend.hpp \
    support/glm/gtx/extend.inl \
    support/glm/gtx/extented_min_max.hpp \
    support/glm/gtx/extented_min_max.inl \
    support/glm/gtx/fast_exponential.hpp \
    support/glm/gtx/fast_exponential.inl \
    support/glm/gtx/fast_square_root.hpp \
    support/glm/gtx/fast_square_root.inl \
    support/glm/gtx/fast_trigonometry.hpp \
    support/glm/gtx/fast_trigonometry.inl \
    support/glm/gtx/gradient_paint.hpp \
    support/glm/gtx/gradient_paint.inl \
    support/glm/gtx/handed_coordinate_space.hpp \
    support/glm/gtx/handed_coordinate_space.inl \
    support/glm/gtx/inertia.hpp \
    support/glm/gtx/inertia.inl \
    support/glm/gtx/int_10_10_10_2.hpp \
    support/glm/gtx/int_10_10_10_2.inl \
    support/glm/gtx/integer.hpp \
    support/glm/gtx/integer.inl \
    support/glm/gtx/intersect.hpp \
    support/glm/gtx/intersect.inl \
    support/glm/gtx/io.hpp \
    support/glm/gtx/io.inl \
    support/glm/gtx/log_base.hpp \
    support/glm/gtx/log_base.inl \
    support/glm/gtx/matrix_cross_product.hpp \
    support/glm/gtx/matrix_cross_product.inl \
    support/glm/gtx/matrix_interpolation.hpp \
    support/glm/gtx/matrix_interpolation.inl \
    support/glm/gtx/matrix_major_storage.hpp \
    support/glm/gtx/matrix_major_storage.inl \
    support/glm/gtx/matrix_operation.hpp \
    support/glm/gtx/matrix_operation.inl \
    support/glm/gtx/matrix_query.hpp \
    support/glm/gtx/matrix_query.inl \
    support/glm/gtx/mixed_product.hpp \
    support/glm/gtx/mixed_product.inl \
    support/glm/gtx/multiple.hpp \
    support/glm/gtx/multiple.inl \
    support/glm/gtx/noise.hpp \
    support/glm/gtx/norm.hpp \
    support/glm/gtx/norm.inl \
    support/glm/gtx/normal.hpp \
    support/glm/gtx/normal.inl \
    support/glm/gtx/normalize_dot.hpp \
    support/glm/gtx/normalize_dot.inl \
    support/glm/gtx/number_precision.hpp \
    support/glm/gtx/number_precision.inl \
    support/glm/gtx/optimum_pow.hpp \
    support/glm/gtx/optimum_pow.inl \
    support/glm/gtx/orthonormalize.hpp \
    support/glm/gtx/orthonormalize.inl \
    support/glm/gtx/perpendicular.hpp \
    support/glm/gtx/perpendicular.inl \
    support/glm/gtx/polar_coordinates.hpp \
    support/glm/gtx/polar_coordinates.inl \
    support/glm/gtx/projection.hpp \
    support/glm/gtx/projection.inl \
    support/glm/gtx/quaternion.hpp \
    support/glm/gtx/quaternion.inl \
    support/glm/gtx/random.hpp \
    support/glm/gtx/raw_data.hpp \
    support/glm/gtx/raw_data.inl \
    support/glm/gtx/reciprocal.hpp \
    support/glm/gtx/rotate_normalized_axis.hpp \
    support/glm/gtx/rotate_normalized_axis.inl \
    support/glm/gtx/rotate_vector.hpp \
    support/glm/gtx/rotate_vector.inl \
    support/glm/gtx/scalar_relational.hpp \
    support/glm/gtx/scalar_relational.inl \
    support/glm/gtx/simd_mat4.hpp \
    support/glm/gtx/simd_mat4.inl \
    support/glm/gtx/simd_quat.hpp \
    support/glm/gtx/simd_quat.inl \
    support/glm/gtx/simd_vec4.hpp \
    support/glm/gtx/simd_vec4.inl \
    support/glm/gtx/spline.hpp \
    support/glm/gtx/spline.inl \
    support/glm/gtx/std_based_type.hpp \
    support/glm/gtx/std_based_type.inl \
    support/glm/gtx/string_cast.hpp \
    support/glm/gtx/string_cast.inl \
    support/glm/gtx/transform.hpp \
    support/glm/gtx/transform.inl \
    support/glm/gtx/transform2.hpp \
    support/glm/gtx/transform2.inl \
    support/glm/gtx/ulp.hpp \
    support/glm/gtx/unsigned_int.hpp \
    support/glm/gtx/unsigned_int.inl \
    support/glm/gtx/vec1.hpp \
    support/glm/gtx/vec1.inl \
    support/glm/gtx/vector_angle.hpp \
    support/glm/gtx/vector_angle.inl \
    support/glm/gtx/vector_query.hpp \
    support/glm/gtx/vector_query.inl \
    support/glm/gtx/wrap.hpp \
    support/glm/gtx/wrap.inl \
    support/glm/integer.hpp \
    support/glm/mat2x2.hpp \
    support/glm/mat2x3.hpp \
    support/glm/mat2x4.hpp \
    support/glm/mat3x2.hpp \
    support/glm/mat3x3.hpp \
    support/glm/mat3x4.hpp \
    support/glm/mat4x2.hpp \
    support/glm/mat4x3.hpp \
    support/glm/mat4x4.hpp \
    support/glm/matrix.hpp \
    support/glm/packing.hpp \
    support/glm/trigonometric.hpp \
    support/glm/vec2.hpp \
    support/glm/vec3.hpp \
    support/glm/vec4.hpp \
    support/glm/vector_relational.hpp \
    support/glm/virtrev/xstream.hpp \
    support/lib/CS123ISceneParser.h \
    support/lib/CS123SceneData.h \
    support/lib/CS123XmlSceneParser.h \
    support/lib/RGBA.h \
    support/lib/ResourceLoader.h \
    support/scenegraph/LSystemTreeScene.h \
    support/scenegraph/OpenGLScene.h \
    support/scenegraph/Scene.h \
    support/scenegraph/ShaderEvolutionTestingScene.h \
    support/scenegraph/ShaderImportScene.h \
    support/scenegraph/SupportCanvas3D.h \
    support/shapes/CircularPlane.h \
    support/shapes/Cone.h \
    support/shapes/Cube.h \
    support/shapes/Cylinder.h \
    support/shapes/HemispherePlane.h \
    support/shapes/Loop.h \
    support/shapes/Rectplane.h \
    support/shapes/Shape.h \
    support/shapes/Sphere.h \
    support/shapes/Surface.h \
    support/shapes/Torus.h \
    ui/shadercodedisplayer.h \
    ui_mainwindow.h \
    glew-1.10.0/include/GL/glew.h \
    lsystems/LSystemUtils.h \
    support/scenegraph/GalleryScene.h

FORMS += ui/mainwindow.ui \
    ui/shadercodedisplayer.ui
INCLUDEPATH += glm ui glew-1.10.0/include
DEPENDPATH += glm ui glew-1.10.0/include

DEFINES += _USE_MATH_DEFINES
DEFINES += TIXML_USE_STL
DEFINES += GLM_SWIZZLE GLM_FORCE_RADIANS
OTHER_FILES += shaders/shader.frag \
    shaders/shader.vert

# Don't add the -pg flag unless you know what you are doing. It makes QThreadPool freeze on Mac OS X
QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS_WARN_ON -= -Wall
QMAKE_CXXFLAGS_WARN_ON += -Waddress -Warray-bounds -Wc++0x-compat -Wchar-subscripts -Wformat\
                          -Wmain -Wmissing-braces -Wparentheses -Wreorder -Wreturn-type \
                          -Wsequence-point -Wsign-compare -Wstrict-overflow=1 -Wswitch \
                          -Wtrigraphs -Wuninitialized -Wunused-label -Wunused-variable \
                          -Wvolatile-register-var -Wno-extra

QMAKE_CXXFLAGS += -g

# QMAKE_CXX_FLAGS_WARN_ON += -Wunknown-pragmas -Wunused-function -Wmain

macx {
    QMAKE_CXXFLAGS_WARN_ON -= -Warray-bounds -Wc++0x-compat
}

RESOURCES += \
    resources.qrc

DISTFILES += \
    shaders/normals/normals.vert \
    shaders/normals/normals.frag \
    shaders/normals/normals.gsh \
    shaders/normals/normalsArrow.gsh \
    shaders/normals/normalsArrow.frag \
    shaders/normals/normalsArrow.vert \
    shaders/shaderevolutionshader.frag \
    shaders/shaderevolutionshader.vert \
    support/glm/CMakeLists.txt \
    xmlScenes/shaderTestingScene.xml
