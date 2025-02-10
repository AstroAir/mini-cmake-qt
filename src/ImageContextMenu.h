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
    void adjustBrightnessRequested();
    void adjustContrastRequested();
    void sharpenRequested();
    void blurRequested();
    void grayscaleRequested();
    void invertColorsRequested();
    void redEyeRemovalRequested();
    void autoEnhanceRequested();
    void addTextRequested();
    void drawRequested();
};

#endif // IMAGECONTEXTMENU_H
