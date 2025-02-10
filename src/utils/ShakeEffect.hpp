#ifndef SHAKEEFFECT_H
#define SHAKEEFFECT_H

#include <QList>
#include <QObject>
#include <QPair>
#include <QPoint>
#include <QPropertyAnimation>
#include <QWidget>
#include <spdlog/spdlog.h>
#include <stdexcept>


/**
 * @brief The ShakeEffect class 实现一个窗口抖动效果。
 */
class ShakeEffect : public QObject {
  Q_OBJECT
public:
  /**
   * @brief 构造函数
   * @param target 要应用抖动效果的窗口控件
   * @param duration 抖动持续时间，单位毫秒（默认500ms）
   * @param shakeDistance 抖动最大偏移距离（默认10像素）
   * @param parent 父对象
   */
  explicit ShakeEffect(QWidget *target, int duration = 500,
                       int shakeDistance = 10, QObject *parent = nullptr)
      : QObject(parent), m_target(target), m_duration(duration),
        m_shakeDistance(shakeDistance) {
    if (!m_target) {
      spdlog::error("ShakeEffect: 无效的目标窗口");
      throw std::invalid_argument("Invalid target widget for ShakeEffect");
    }
  }

  /**
   * @brief 启动抖动效果
   */
  void start() {
    try {
      m_originalPos = m_target->pos();

      // 创建动画对象
      auto animation = new QPropertyAnimation(m_target, "pos", this);
      animation->setDuration(m_duration);
      animation->setLoopCount(1);

      // 设置动画关键帧，这里使用了一组预定义的位移实现抖动效果
      QList<QPair<int, QPoint>> keyFrames = {
          {0, m_originalPos},
          {10, m_originalPos + QPoint(-m_shakeDistance, 0)},
          {20, m_originalPos + QPoint(m_shakeDistance, 0)},
          {30, m_originalPos + QPoint(0, -m_shakeDistance)},
          {40, m_originalPos + QPoint(0, m_shakeDistance)},
          {50, m_originalPos + QPoint(-m_shakeDistance, 0)},
          {60, m_originalPos + QPoint(m_shakeDistance, 0)},
          {70, m_originalPos + QPoint(0, m_shakeDistance)},
          {80, m_originalPos + QPoint(0, -m_shakeDistance)},
          {90, m_originalPos},
          {100, m_originalPos}};

      for (const auto &keyFrame : keyFrames) {
        double step = keyFrame.first / 100.0;
        animation->setKeyValueAt(step, keyFrame.second);
      }

      animation->start(QAbstractAnimation::DeleteWhenStopped);
      spdlog::info("ShakeEffect: 开始对窗口 '{}' 进行抖动",
                   m_target->objectName().toStdString());
    } catch (const std::exception &e) {
      spdlog::error("ShakeEffect: 启动抖动时发生异常: {}", e.what());
    }
  }

private:
  QWidget *m_target;
  int m_duration;
  int m_shakeDistance;
  QPoint m_originalPos;
};

#endif // SHAKEEFFECT_H