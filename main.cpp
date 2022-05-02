#include <QApplication>
#include "mainwindow.h"


#ifdef CUDA_FLUID
extern "C" void get_deviceinfo();
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;
    bool startFullscreen = false;
    w.show();

    #ifdef CUDA_FLUID
    get_deviceinfo();
    #endif

    if (startFullscreen) {
        // We cannot use w.showFullscreen() here because on Linux that creates the
        // window behind all other windows, so we have to set it to fullscreen after
        // it has been shown.
        w.setWindowState(w.windowState() | Qt::WindowFullScreen);
        w.setWindowIcon(QIcon(":/icon/killWithFire.png"));
    }

    return app.exec();
}
