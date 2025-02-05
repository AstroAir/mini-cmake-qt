#ifndef IMAGECONTEXTMENU_H
#define IMAGECONTEXTMENU_H

#include <QMenu>

class ImageContextMenu : public QMenu {
    Q_OBJECT
public:
    explicit ImageContextMenu(QWidget *parent = nullptr);

signals:
    void zoomInRequested();
    void zoomOutRequested();
    void rotateLeftRequested();
    void rotateRightRequested();
    void toggleFullscreenRequested();
    void copyRequested();
    void saveRequested();
    void cropRequested();
    void flipHorizontalRequested();
    void flipVerticalRequested();
    void resetRequested();
};

#endif // IMAGECONTEXTMENU_H
