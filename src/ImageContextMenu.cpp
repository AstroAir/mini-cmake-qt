#include "ImageContextMenu.h"
#include <QAction>

ImageContextMenu::ImageContextMenu(QWidget *parent) : QMenu(parent) {
    // 基本操作
    addAction(tr("放大"), this, &ImageContextMenu::zoomInRequested);
    addAction(tr("缩小"), this, &ImageContextMenu::zoomOutRequested);
    addSeparator();
    
    // 旋转操作
    addAction(tr("向左旋转"), this, &ImageContextMenu::rotateLeftRequested);
    addAction(tr("向右旋转"), this, &ImageContextMenu::rotateRightRequested);
    addSeparator();
    
    // 视图操作
    addAction(tr("全屏"), this, &ImageContextMenu::toggleFullscreenRequested);
    addSeparator();
    
    // 编辑操作
    addAction(tr("复制"), this, &ImageContextMenu::copyRequested);
    addAction(tr("保存"), this, &ImageContextMenu::saveRequested);
    addSeparator();
    
    // 图像处理操作
    addAction(tr("裁剪"), this, &ImageContextMenu::cropRequested);
    addAction(tr("水平翻转"), this, &ImageContextMenu::flipHorizontalRequested);
    addAction(tr("垂直翻转"), this, &ImageContextMenu::flipVerticalRequested);
    addSeparator();
    
    // 重置操作
    addAction(tr("重置"), this, &ImageContextMenu::resetRequested);
}
