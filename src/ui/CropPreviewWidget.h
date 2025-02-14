#pragma once

#include "../image/Crop.h"
#include <QWidget>
#include <opencv2/opencv.hpp>

class CropPreviewWidget : public QWidget {
  Q_OBJECT

public:
  explicit CropPreviewWidget(QWidget *parent = nullptr);
  void setImage(const cv::Mat &image);
  void setImage(const QImage &image); // 新增QImage重载
  void resetView();  // 新增resetView方法
  void setStrategy(const CropStrategy &strategy);
  CropStrategy getCurrentStrategy() const;
  void zoomIn();
  void zoomOut();
  void fitToView();
  void setZoom(float zoom);

signals:
  void strategyChanged();
  void errorOccurred(const QString &error);
  void mousePositionChanged(const QPoint &pos);
  void zoomChanged(float zoom);

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void enterEvent(QEnterEvent *event) override;
  void leaveEvent(QEvent *event) override;

private:
  void drawCropShape(QPainter &painter);
  QPointF mapToImage(const QPoint &pos) const;
  QPoint mapFromImage(const cv::Point &pos) const;
  void updateCropPoints(const QPoint &pos);
  void handleDragOperation(const QPoint &pos);
  bool validateOperation();
  void emitError(const QString &error);
  void updateCursor();

  // 控制点类型
  enum class ControlPoint {
    None = -1,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
    Rotation
  };

  enum class DragMode { None, Move, Resize, Rotate } dragMode = DragMode::None;

  // 辅助方法
  DragMode determineDragMode(const QPoint &pos);
  void updateRectangleDrag(cv::Rect &rect, const cv::Point &pos);
  void updateCircleDrag(CircleCrop &circle, const cv::Point &pos);
  void updateEllipseDrag(EllipseCrop &ellipse, const cv::Point &pos);
  void updatePolygonDrag(std::vector<cv::Point> &points, const cv::Point &pos);
  ControlPoint hitTest(const QPoint &pos) const;
  QRect getControlRect(const QPoint &pt, int size = 8) const;

  // 添加成员变量
  ControlPoint activeControl = ControlPoint::None;
  bool isRotating = false;
  double startAngle = 0.0;
  QPoint rotationCenter;
  int controlPointSize = 8;

  cv::Mat image;
  QImage qtImage;
  CropStrategy currentStrategy;
  bool isDragging = false;
  int activePoint = -1;
  QPointF lastPos;
  float scale = 1.0f;
  float zoomLevel = 1.0f;
  float minZoom = 0.1f;
  float maxZoom = 5.0f;
  QPointF viewCenter;

  QCursor defaultCursor;
  bool isHovered = false;
};
