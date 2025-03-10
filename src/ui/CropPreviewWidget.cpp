#include "CropPreviewWidget.h"
#include <QMouseEvent>
#include <QPainter>

CropPreviewWidget::CropPreviewWidget(QWidget *parent) : QWidget(parent) {
  setMouseTracking(true);
  setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void CropPreviewWidget::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);

  // 绘制背景图
  if (!qtImage.isNull()) {
    // 计算缩放后的图像位置，使其居中显示
    QSize scaledSize = qtImage.size() * scale;
    QPoint pos((width() - scaledSize.width()) / 2,
               (height() - scaledSize.height()) / 2);

    painter.translate(pos);
    painter.scale(scale, scale);
    painter.drawImage(0, 0, qtImage);
    painter.resetTransform();
  }

  // 绘制网格
  if (drawGrid) {
    drawGridLines(painter);
  }

  // 绘制裁切形状
  drawCropShape(painter);
}

void CropPreviewWidget::setImage(const cv::Mat &img) {
  if (img.empty())
    return;

  image = img.clone();
  cv::Mat rgb;
  if (img.channels() == 1) {
    cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
  } else {
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  }

  qtImage =
      QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888)
          .copy();

  scale = std::min(float(width()) / image.cols, float(height()) / image.rows);

  update();
}

void CropPreviewWidget::setImage(const QImage &image) {
    qtImage = image;
    // 将QImage转换为cv::Mat
    if (image.format() == QImage::Format_RGB32 || 
        image.format() == QImage::Format_ARGB32 ||
        image.format() == QImage::Format_ARGB32_Premultiplied) {
        cv::Mat temp(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(temp, this->image, cv::COLOR_BGRA2BGR);
    } else {
        QImage converted = image.convertToFormat(QImage::Format_RGB888);
        cv::Mat temp(converted.height(), converted.width(), CV_8UC3, (void*)converted.constBits(), converted.bytesPerLine());
        cv::cvtColor(temp, this->image, cv::COLOR_RGB2BGR);
    }
    update();
}

void CropPreviewWidget::resetView() {
    scale = 1.0f;
    zoomLevel = 1.0f;
    viewCenter = QPointF(0, 0);
    update();
    emit zoomChanged(zoomLevel);
}

void CropPreviewWidget::setStrategy(const CropStrategy &strategy) {
  currentStrategy = strategy;
  update();
}

void CropPreviewWidget::drawCropShape(QPainter &painter) {
  if (image.empty())
    return;

  QPen pen(Qt::white);
  pen.setWidth(2);
  painter.setPen(pen);

  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, cv::Rect>) {
          QRect rect(mapFromImage({arg.x, arg.y}),
                     mapFromImage({arg.x + arg.width, arg.y + arg.height}));
          painter.drawRect(rect);
        } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
          QPolygon poly;
          for (const auto &pt : arg) {
            poly << mapFromImage(pt);
          }
          painter.drawPolygon(poly);
        } else if constexpr (std::is_same_v<T, CircleCrop>) {
          QPoint center = mapFromImage(arg.center);
          int radius = int(arg.radius * scale);
          painter.drawEllipse(center, radius, radius);
        } else if constexpr (std::is_same_v<T, EllipseCrop>) {
          QPoint center = mapFromImage(arg.center);
          QSize axes(int(arg.axes.width * scale), int(arg.axes.height * scale));
          painter.save();
          painter.translate(center);
          painter.rotate(arg.angle);
          painter.drawEllipse(QPoint(0, 0), axes.width(), axes.height());
          painter.restore();
        }
      },
      currentStrategy);
}

QPointF CropPreviewWidget::mapToImage(const QPoint &pos) const {
  return QPointF(pos.x() / scale, pos.y() / scale);
}

QPoint CropPreviewWidget::mapFromImage(const cv::Point &pos) const {
  return QPoint(int(pos.x * scale), int(pos.y * scale));
}

void CropPreviewWidget::mousePressEvent(QMouseEvent *event) {
  if (!validateOperation())
    return;

  isDragging = true;
  lastPos = event->pos();
  activeControl = hitTest(event->pos());
  dragMode = determineDragMode(event->pos());

  if (dragMode == DragMode::Rotate) {
    isRotating = true;
    // 计算旋转起始角度
    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, EllipseCrop>) {
            rotationCenter = mapFromImage(arg.center);
            QLineF line(rotationCenter, lastPos);
            startAngle = line.angle();
          }
        },
        currentStrategy);
  }

  updateCursor();
  event->accept();
}

void CropPreviewWidget::mouseMoveEvent(QMouseEvent *event) {
  if (!validateOperation())
    return;

  QPoint pos = event->pos();
  emit mousePositionChanged(pos);

  if (isDragging) {
    handleDragOperation(pos);
  } else {
    // 更新鼠标样式
    dragMode = determineDragMode(pos);
    updateCursor();
  }
}

void CropPreviewWidget::mouseReleaseEvent(QMouseEvent *event) {
  isDragging = false;
  activePoint = -1;
}

void CropPreviewWidget::updateCropPoints(const QPoint &pos) {
  auto imgPos = mapToImage(pos);
  cv::Point cvPos(int(imgPos.x()), int(imgPos.y()));

  std::visit(
      [&](auto &&arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, cv::Rect>) {
          // 更新矩形位置或大小
          if (activePoint >= 0) {
            // 处理控制点拖拽
          } else {
            // 处理整体移动
            arg.x = std::clamp(cvPos.x, 0, image.cols - arg.width);
            arg.y = std::clamp(cvPos.y, 0, image.rows - arg.height);
          }
        }
        // 处理其他形状的更新...
      },
      currentStrategy);
}

CropStrategy CropPreviewWidget::getCurrentStrategy() const {
  return currentStrategy;
}

void CropPreviewWidget::zoomIn() { setZoom(zoomLevel * 1.2f); }

void CropPreviewWidget::zoomOut() { setZoom(zoomLevel / 1.2f); }

void CropPreviewWidget::fitToView() {
  if (image.empty())
    return;

  float scaleX = float(width()) / image.cols;
  float scaleY = float(height()) / image.rows;
  setZoom(std::min(scaleX, scaleY));
}

void CropPreviewWidget::setZoom(float zoom) {
  zoomLevel = std::clamp(zoom, minZoom, maxZoom);
  scale = zoomLevel;
  update();
}

void CropPreviewWidget::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() & Qt::ControlModifier) {
    // Ctrl + 滚轮实现缩放
    float delta = event->angleDelta().y() / 120.f;
    float newZoom = zoomLevel * (1.0f + delta * 0.1f);
    setZoom(newZoom);
    event->accept();
  } else {
    event->ignore();
  }
}

void CropPreviewWidget::updateCursor() {
  if (!isHovered) {
    setCursor(defaultCursor);
    return;
  }

  switch (dragMode) {
  case DragMode::Move:
    setCursor(Qt::SizeAllCursor);
    break;
  case DragMode::Resize:
    setCursor(Qt::SizeFDiagCursor);
    break;
  case DragMode::Rotate:
    setCursor(Qt::CrossCursor);
    break;
  default:
    setCursor(defaultCursor);
  }
}

void CropPreviewWidget::handleDragOperation(const QPoint &pos) {
  if (!validateOperation())
    return;

  try {
    auto imgPos = mapToImage(pos);
    cv::Point cvPos(int(imgPos.x()), int(imgPos.y()));

    std::visit(
        [&](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, cv::Rect>) {
            updateRectangleDrag(arg, cvPos);
          } else if constexpr (std::is_same_v<T, CircleCrop>) {
            updateCircleDrag(arg, cvPos);
          } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            updateEllipseDrag(arg, cvPos);
          } else if constexpr (std::is_same_v<T, std::vector<cv::Point>>) {
            updatePolygonDrag(arg, cvPos);
          }
        },
        currentStrategy);

    emit strategyChanged();
    update();
  } catch (const std::exception &e) {
    emitError(QString::fromUtf8(e.what()));
  }
}

bool CropPreviewWidget::validateOperation() {
  if (image.empty()) {
    emitError(tr("没有可用的图像"));
    return false;
  }
  return true;
}

void CropPreviewWidget::emitError(const QString &error) {
  emit errorOccurred(error);
}

CropPreviewWidget::DragMode
CropPreviewWidget::determineDragMode(const QPoint &pos) {
  ControlPoint control = hitTest(pos);
  if (control != ControlPoint::None) {
    // 如果点击在控制点上
    if (control == ControlPoint::Rotation) {
      return DragMode::Rotate;
    }
    return DragMode::Resize;
  }

  // 检查是否在裁剪区域内
  bool inShape = std::visit(
      [&](auto &&arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, cv::Rect>) {
          QRect rect(mapFromImage({arg.x, arg.y}),
                     mapFromImage({arg.x + arg.width, arg.y + arg.height}));
          return rect.contains(pos);
        } else if constexpr (std::is_same_v<T, CircleCrop>) {
          QPoint center = mapFromImage(arg.center);
          int radius = int(arg.radius * scale);
          return QLineF(center, pos).length() <= radius;
        }
        return false;
      },
      currentStrategy);

  return inShape ? DragMode::Move : DragMode::None;
}

void CropPreviewWidget::updateRectangleDrag(cv::Rect &rect,
                                            const cv::Point &pos) {
  QPoint delta = QPoint(pos.x * scale, pos.y * scale) - lastPos.toPoint();

  switch (activeControl) {
  case ControlPoint::TopLeft:
    rect.x += delta.x() / scale;
    rect.y += delta.y() / scale;
    rect.width -= delta.x() / scale;
    rect.height -= delta.y() / scale;
    break;
  case ControlPoint::TopRight:
    rect.y += delta.y() / scale;
    rect.width += delta.x() / scale;
    rect.height -= delta.y() / scale;
    break;
  case ControlPoint::BottomLeft:
    rect.x += delta.x() / scale;
    rect.width -= delta.x() / scale;
    rect.height += delta.y() / scale;
    break;
  case ControlPoint::BottomRight:
    rect.width += delta.x() / scale;
    rect.height += delta.y() / scale;
    break;
  case ControlPoint::Top:
    rect.y += delta.y() / scale;
    rect.height -= delta.y() / scale;
    break;
  case ControlPoint::Bottom:
    rect.height += delta.y() / scale;
    break;
  case ControlPoint::Left:
    rect.x += delta.x() / scale;
    rect.width -= delta.x() / scale;
    break;
  case ControlPoint::Right:
    rect.width += delta.x() / scale;
    break;
  default:
    // 移动整个矩形
    rect.x += delta.x() / scale;
    rect.y += delta.y() / scale;
    break;
  }

  // 确保矩形不会变成负数
  if (rect.width < 10)
    rect.width = 10;
  if (rect.height < 10)
    rect.height = 10;

  // 限制在图像范围内
  rect.x = std::clamp(rect.x, 0, image.cols - rect.width);
  rect.y = std::clamp(rect.y, 0, image.rows - rect.height);
}

void CropPreviewWidget::updateCircleDrag(CircleCrop &circle,
                                         const cv::Point &pos) {
  QPoint delta = QPoint(pos.x * scale, pos.y * scale) - lastPos.toPoint();

  if (activeControl != ControlPoint::None) {
    // 调整半径
    QPoint center = mapFromImage(circle.center);
    float newRadius = QLineF(center, mapFromImage(pos)).length() / scale;
    circle.radius =
        std::clamp(int(newRadius), 10, std::min(image.cols, image.rows) / 2);
  } else {
    // 移动圆心
    circle.center.x += delta.x() / scale;
    circle.center.y += delta.y() / scale;

    // 限制在图像范围内
    circle.center.x =
        std::clamp(circle.center.x, circle.radius, image.cols - circle.radius);
    circle.center.y =
        std::clamp(circle.center.y, circle.radius, image.rows - circle.radius);
  }
}

CropPreviewWidget::ControlPoint
CropPreviewWidget::hitTest(const QPoint &pos) const {
  return std::visit(
      [&](auto &&arg) -> ControlPoint {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, cv::Rect>) {
          QRect rect(mapFromImage({arg.x, arg.y}),
                     mapFromImage({arg.x + arg.width, arg.y + arg.height}));

          // 检查八个控制点
          if (getControlRect(rect.topLeft()).contains(pos))
            return ControlPoint::TopLeft;
          if (getControlRect(rect.topRight()).contains(pos))
            return ControlPoint::TopRight;
          if (getControlRect(rect.bottomLeft()).contains(pos))
            return ControlPoint::BottomLeft;
          if (getControlRect(rect.bottomRight()).contains(pos))
            return ControlPoint::BottomRight;
          if (getControlRect(QPoint(rect.center().x(), rect.top()))
                  .contains(pos))
            return ControlPoint::Top;
          if (getControlRect(QPoint(rect.center().x(), rect.bottom()))
                  .contains(pos))
            return ControlPoint::Bottom;
          if (getControlRect(QPoint(rect.left(), rect.center().y()))
                  .contains(pos))
            return ControlPoint::Left;
          if (getControlRect(QPoint(rect.right(), rect.center().y()))
                  .contains(pos))
            return ControlPoint::Right;
        } else if constexpr (std::is_same_v<T, CircleCrop>) {
          QPoint center = mapFromImage(arg.center);
          int radius = int(arg.radius * scale);
          QLineF line(center, pos);
          if (std::abs(line.length() - radius) < controlPointSize)
            return ControlPoint::Right; // 使用Right表示圆的边界点
        }
        return ControlPoint::None;
      },
      currentStrategy);
}

QRect CropPreviewWidget::getControlRect(const QPoint &pt, int size) const {
  return QRect(pt.x() - size / 2, pt.y() - size / 2, size, size);
}

void CropPreviewWidget::enterEvent(QEnterEvent *event) {
  isHovered = true;
  updateCursor();
  QWidget::enterEvent(event);
}

void CropPreviewWidget::leaveEvent(QEvent *event) {
  isHovered = false;
  updateCursor();
  QWidget::leaveEvent(event);
}

void CropPreviewWidget::updateEllipseDrag(EllipseCrop &ellipse, const cv::Point &pos) {
  QPoint delta = QPoint(pos.x * scale, pos.y * scale) - lastPos.toPoint();

  if (isRotating) {
    // 处理旋转
    QPointF currentPos(pos.x * scale, pos.y * scale);
    QLineF line(rotationCenter, currentPos);
    double currentAngle = line.angle();
    double angleDelta = currentAngle - startAngle;
    ellipse.angle = std::fmod(ellipse.angle + angleDelta + 360.0, 360.0);
    startAngle = currentAngle;
  } else if (activeControl != ControlPoint::None) {
    // 调整大小
    float dx = delta.x() / scale;
    float dy = delta.y() / scale;
    
    switch (activeControl) {
      case ControlPoint::Right:
        ellipse.axes.width += dx;
        break;
      case ControlPoint::Bottom:
        ellipse.axes.height += dy;
        break;
      default:
        break;
    }
    
    // 确保最小尺寸
    ellipse.axes.width = std::max(ellipse.axes.width, 10);
    ellipse.axes.height = std::max(ellipse.axes.height, 10);
  } else {
    // 移动中心点
    ellipse.center.x += delta.x() / scale;
    ellipse.center.y += delta.y() / scale;

    // 限制在图像范围内
    ellipse.center.x = std::clamp(ellipse.center.x, 
      int(ellipse.axes.width), image.cols - int(ellipse.axes.width));
    ellipse.center.y = std::clamp(ellipse.center.y, 
      int(ellipse.axes.height), image.rows - int(ellipse.axes.height));
  }
}

void CropPreviewWidget::updatePolygonDrag(std::vector<cv::Point> &points, const cv::Point &pos) {
  QPoint delta = QPoint(pos.x * scale, pos.y * scale) - lastPos.toPoint();
  
  if (activePoint >= 0 && activePoint < points.size()) {
    // 移动选中的点
    points[activePoint].x += delta.x() / scale;
    points[activePoint].y += delta.y() / scale;
    
    // 限制在图像范围内
    points[activePoint].x = std::clamp(points[activePoint].x, 0, image.cols);
    points[activePoint].y = std::clamp(points[activePoint].y, 0, image.rows);
  } else {
    // 移动整个多边形
    for (auto& point : points) {
      point.x += delta.x() / scale;
      point.y += delta.y() / scale;
    }
    
    // 检查边界
    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    
    for (const auto& point : points) {
      minX = std::min(minX, point.x);
      minY = std::min(minY, point.y);
      maxX = std::max(maxX, point.x);
      maxY = std::max(maxY, point.y);
    }
    
    if (minX < 0 || minY < 0 || maxX >= image.cols || maxY >= image.rows) {
      // 如果超出边界，恢复移动
      for (auto& point : points) {
        point.x -= delta.x() / scale;
        point.y -= delta.y() / scale;
      }
    }
  }
}

void CropPreviewWidget::setAspectRatioLocked(bool locked) {
    if (aspectRatioLocked != locked) {
        aspectRatioLocked = locked;
        if (locked) {
            enforceAspectRatio();
        }
        update();
    }
}

void CropPreviewWidget::setGridVisible(bool visible) {
    if (drawGrid != visible) {
        drawGrid = visible;
        update();
    }
}

void CropPreviewWidget::setAspectRatio(double ratio) {
    if (aspectRatio != ratio) {
        aspectRatio = ratio;
        if (aspectRatioLocked) {
            enforceAspectRatio();
        }
        update();
    }
}

void CropPreviewWidget::enforceAspectRatio() {
    std::visit([this](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, cv::Rect>) {
            // 调整矩形以匹配宽高比
            double currentRatio = static_cast<double>(arg.width) / arg.height;
            if (std::abs(currentRatio - aspectRatio) > 0.001) {
                if (currentRatio > aspectRatio) {
                    // 需要减小宽度
                    int newWidth = static_cast<int>(arg.height * aspectRatio);
                    arg.x += (arg.width - newWidth) / 2;
                    arg.width = newWidth;
                } else {
                    // 需要减小高度
                    int newHeight = static_cast<int>(arg.width / aspectRatio);
                    arg.y += (arg.height - newHeight) / 2;
                    arg.height = newHeight;
                }
            }
        } else if constexpr (std::is_same_v<T, EllipseCrop>) {
            // 调整椭圆以匹配宽高比
            double currentRatio = static_cast<double>(arg.axes.width) / arg.axes.height;
            if (std::abs(currentRatio - aspectRatio) > 0.001) {
                if (currentRatio > aspectRatio) {
                    arg.axes.width = static_cast<int>(arg.axes.height * aspectRatio);
                } else {
                    arg.axes.height = static_cast<int>(arg.axes.width / aspectRatio);
                }
            }
        }
    }, currentStrategy);
}

void CropPreviewWidget::drawGridLines(QPainter& painter) {
    if (!drawGrid) return;

    painter.save();
    
    // 设置网格线的样式
    QPen gridPen(Qt::white);
    gridPen.setStyle(Qt::DashLine);
    gridPen.setWidth(1);
    painter.setPen(gridPen);
    
    // 计算网格大小
    int gridSize = 50;  // 可以根据需要调整网格大小
    
    // 绘制垂直线
    for (int x = gridSize; x < width(); x += gridSize) {
        painter.drawLine(x, 0, x, height());
    }
    
    // 绘制水平线
    for (int y = gridSize; y < height(); y += gridSize) {
        painter.drawLine(0, y, width(), y);
    }
    
    painter.restore();
}
