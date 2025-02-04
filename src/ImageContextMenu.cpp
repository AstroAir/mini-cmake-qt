#include "ImageContextMenu.h"
#include <QAction>

ImageContextMenu::ImageContextMenu(QWidget *parent) : QMenu(parent) {
  addAction("放大", this, [this](){ emit zoomInRequested(); });
  addAction("缩小", this, [this](){ emit zoomOutRequested(); });
  addSeparator();
  addAction("向左旋转", this, [this](){ emit rotateLeftRequested(); });
  addAction("向右旋转", this, [this](){ emit rotateRightRequested(); });
  addSeparator();
  addAction("全屏切换", this, [this](){ emit toggleFullscreenRequested(); });
  addSeparator();
  addAction("复制", this, [this](){ emit copyRequested(); });
  addAction("保存", this, [this](){ emit saveRequested(); });
  // 新增操作菜单项
  addSeparator();
  addAction("裁剪", this, [this](){ emit cropRequested(); });
  addAction("水平翻转", this, [this](){ emit flipHorizontalRequested(); });
  addAction("垂直翻转", this, [this](){ emit flipVerticalRequested(); });
  addAction("重置图像", this, [this](){ emit resetRequested(); });
}
