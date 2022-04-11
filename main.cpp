#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;
    bool startFullscreen = false;
    w.show();

    if (startFullscreen) {
        // We cannot use w.showFullscreen() here because on Linux that creates the
        // window behind all other windows, so we have to set it to fullscreen after
        // it has been shown.
        w.setWindowState(w.windowState() | Qt::WindowFullScreen);
    }

    return app.exec();
}
