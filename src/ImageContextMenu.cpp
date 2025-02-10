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

  // 图像调整
  QMenu *adjustMenu = addMenu(tr("图像调整"));
  adjustMenu->addAction(tr("亮度"), this,
                        &ImageContextMenu::adjustBrightnessRequested);
  adjustMenu->addAction(tr("对比度"), this,
                        &ImageContextMenu::adjustContrastRequested);
  adjustMenu->addAction(tr("锐化"), this, &ImageContextMenu::sharpenRequested);
  adjustMenu->addAction(tr("模糊"), this, &ImageContextMenu::blurRequested);

  // 滤镜效果
  QMenu *filterMenu = addMenu(tr("滤镜效果"));
  filterMenu->addAction(tr("灰度"), this,
                        &ImageContextMenu::grayscaleRequested);
  filterMenu->addAction(tr("反色"), this,
                        &ImageContextMenu::invertColorsRequested);
  filterMenu->addAction(tr("红眼消除"), this,
                        &ImageContextMenu::redEyeRemovalRequested);
  filterMenu->addAction(tr("自动优化"), this,
                        &ImageContextMenu::autoEnhanceRequested);

  // 编辑工具
  QMenu *editMenu = addMenu(tr("编辑工具"));
  editMenu->addAction(tr("添加文字"), this,
                      &ImageContextMenu::addTextRequested);
  editMenu->addAction(tr("绘图"), this, &ImageContextMenu::drawRequested);

  addSeparator();

  // 重置操作
  addAction(tr("重置"), this, &ImageContextMenu::resetRequested);
}
