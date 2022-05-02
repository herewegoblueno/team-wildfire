#-------------------------------------------------
# Define output directories
DESTDIR = ../bin
CUDA_OBJECTS_DIR = ./

#OTHER_FILES += \
#    $$PWD/jacobi.cu \
#    $$PWD/base_op.cu
CUDA_SOURCES += \
    $$PWD/jacobi.cu \
    $$PWD/base_op.cu \
    $$PWD/info.cu


# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

win32 {
    CONFIG += console
    DEFINES += CUDA_FLUID

    # CUDA settings
    CUDA_DIR = "D:/DL Tools/NVIDIA Corporation/NVIDIA GPU Computing Toolkit/CUDA/v11.1"
    SYSTEM_NAME = x64                 # Depending on your system either 'Win32', 'x64', or 'Win64'
    SYSTEM_TYPE = 64                    # '32' or '64', depending on your system
    CUDA_ARCH = compute_75
    CUDA_CODE = sm_75               # Type of CUDA architecture
    NVCC_OPTIONS = --use_fast_math


    # include paths
    # include paths
    INCLUDEPATH += $$CUDA_DIR/include \
                   $$CUDA_DIR/common/inc \
                   $$CUDA_DIR/../shared/inc

    # library directories
    QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME \
                    $$CUDA_DIR/common/lib/$$SYSTEM_NAME \
                    $$CUDA_DIR/../shared/lib/$$SYSTEM_NAME

    # The following makes sure all path names (which often include spaces) are put between quotation marks
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

    # Add the necessary libraries
    CUDA_LIB_NAMES = cudart_static kernel32 user32 gdi32 winspool comdlg32 \
                     advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32 \
                     #freeglut glew32

    for(lib, CUDA_LIB_NAMES) {
        CUDA_LIBS += -l$$lib
    }
    LIBS += $$CUDA_LIBS

    # Configuration of the Cuda compiler
    CONFIG(debug, debug|release) {
        # Debug mode
        cuda_d.input = CUDA_SOURCES
        cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
        cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                          --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                          --compile -cudart static -g -DWIN32 -D_MBCS \
                          -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                          -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                          -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda_d.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda_d
    }
    else {
        # Release mode
        cuda.input = CUDA_SOURCES
        cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
        cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                        --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                        --compile -cudart static -DWIN32 -D_MBCS \
                        -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                        -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                        -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
        cuda.dependency_type = TYPE_C
        QMAKE_EXTRA_COMPILERS += cuda
    }

}

HEADERS += \
    $$PWD/info.h

