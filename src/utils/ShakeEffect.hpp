#ifndef SHAKEEFFECT_H
#define SHAKEEFFECT_H

#include <QEasingCurve>
#include <QList>
#include <QObject>
#include <QPair>
#include <QPoint>
#include <QPropertyAnimation>
#include <QRandomGenerator>
#include <QTime>
#include <QTimer>
#include <QWidget>

/**
 * @brief 抖动效果类型枚举
 */
enum class ShakeType {
  Horizontal, // 水平抖动
  Vertical,   // 垂直抖动
  Circular,   // 圆形抖动
  Random,     // 随机抖动
  Elastic     // 弹性抖动
};

/**
 * @brief 抖动效果配置结构体
 */
struct ShakeConfig {
  int duration = 500;                                       // 持续时间(ms)
  int shakeDistance = 10;                                   // 抖动距离(像素)
  int frameCount = 10;                                      // 关键帧数量
  float dampingRatio = 0.7f;                                // 阻尼比例
  ShakeType type = ShakeType::Horizontal;                   // 抖动类型
  QEasingCurve::Type easingType = QEasingCurve::OutElastic; // 缓动类型
  bool autoReverse = false;                                 // 是否自动反向
  int loopCount = 1;                                        // 循环次数
};

/**
 * @brief The ShakeEffect class 实现一个高度可定制的窗口抖动效果
 */
class ShakeEffect : public QObject {
  Q_OBJECT
public:
  explicit ShakeEffect(QWidget *target,
                       const ShakeConfig &config = ShakeConfig(),
                       QObject *parent = nullptr);

  // 开始抖动
  void start();

  // 停止抖动
  void stop();

  // 暂停抖动
  void pause();

  // 恢复抖动
  void resume();

  // 更新配置
  void updateConfig(const ShakeConfig &newConfig);

signals:
  void effectStarted();
  void effectStopped();
  void effectPaused();
  void effectResumed();
  void effectCompleted();

private:
  void initializeAnimation();

  void generateKeyFrames();

  void generateHorizontalShake(QList<QPair<qreal, QPoint>> &keyFrames);

  void generateVerticalShake(QList<QPair<qreal, QPoint>> &keyFrames);

  void generateCircularShake(QList<QPair<qreal, QPoint>> &keyFrames);

  void generateRandomShake(QList<QPair<qreal, QPoint>> &keyFrames);

  void generateElasticShake(QList<QPair<qreal, QPoint>> &keyFrames);

private slots:
  void onAnimationFinished();

private:
  QWidget *m_target;
  ShakeConfig m_config;
  QPropertyAnimation *m_animation = nullptr;
  QPoint m_originalPos;
  bool m_isPlaying;
  bool m_isPaused;
};

#endif // SHAKEEFFECT_H