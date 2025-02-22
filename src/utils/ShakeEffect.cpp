#include "ShakeEffect.hpp"

#include <spdlog/spdlog.h>

ShakeEffect::ShakeEffect(QWidget *target, const ShakeConfig &config,
                         QObject *parent)
    : QObject(parent), m_target(target), m_config(config), m_isPlaying(false),
      m_isPaused(false) {

  if (!m_target) {
    spdlog::error("ShakeEffect: 无效的目标窗口");
    throw std::invalid_argument("Invalid target widget");
  }

  initializeAnimation();
  spdlog::debug(
      "ShakeEffect: 使用配置初始化完成 - 类型:{}, 持续时间:{}, 距离:{}",
      static_cast<int>(config.type), config.duration, config.shakeDistance);
}

// 开始抖动
void ShakeEffect::start() {
  if (m_isPlaying) {
    spdlog::warn("ShakeEffect: 动画已在运行中");
    return;
  }

  try {
    m_originalPos = m_target->pos();
    m_animation->start();
    m_isPlaying = true;
    emit effectStarted();
    spdlog::info("ShakeEffect: 开始对窗口 '{}' 进行抖动",
                 m_target->objectName().toStdString());
  } catch (const std::exception &e) {
    spdlog::error("ShakeEffect: 启动失败: {}", e.what());
  }
}

// 停止抖动
void ShakeEffect::stop() {
  if (!m_isPlaying)
    return;

  m_animation->stop();
  m_target->move(m_originalPos);
  m_isPlaying = false;
  m_isPaused = false;
  emit effectStopped();
  spdlog::info("ShakeEffect: 停止抖动");
}

// 暂停抖动
void ShakeEffect::pause() {
  if (!m_isPlaying || m_isPaused)
    return;

  m_animation->pause();
  m_isPaused = true;
  emit effectPaused();
  spdlog::debug("ShakeEffect: 暂停抖动");
}

// 恢复抖动
void ShakeEffect::resume() {
  if (!m_isPaused)
    return;

  m_animation->resume();
  m_isPaused = false;
  emit effectResumed();
  spdlog::debug("ShakeEffect: 恢复抖动");
}

// 更新配置
void ShakeEffect::updateConfig(const ShakeConfig &newConfig) {
  m_config = newConfig;
  initializeAnimation();
  spdlog::info("ShakeEffect: 更新配置");
}

void ShakeEffect::initializeAnimation() {
  if (m_animation) {
    m_animation->deleteLater();
  }

  m_animation = new QPropertyAnimation(m_target, "pos", this);
  m_animation->setDuration(m_config.duration);
  m_animation->setLoopCount(m_config.loopCount);
  m_animation->setEasingCurve(m_config.easingType);

  connect(m_animation, &QPropertyAnimation::finished, this,
          &ShakeEffect::onAnimationFinished);

  generateKeyFrames();
}

void ShakeEffect::generateKeyFrames() {
  QList<QPair<qreal, QPoint>> keyFrames;
  const double PI = 3.14159265359;

  switch (m_config.type) {
  case ShakeType::Horizontal:
    generateHorizontalShake(keyFrames);
    break;
  case ShakeType::Vertical:
    generateVerticalShake(keyFrames);
    break;
  case ShakeType::Circular:
    generateCircularShake(keyFrames);
    break;
  case ShakeType::Random:
    generateRandomShake(keyFrames);
    break;
  case ShakeType::Elastic:
    generateElasticShake(keyFrames);
    break;
  }

  // 设置关键帧
  for (const auto &frame : keyFrames) {
    m_animation->setKeyValueAt(frame.first, frame.second);
  }
}

void ShakeEffect::generateHorizontalShake(
    QList<QPair<qreal, QPoint>> &keyFrames) {
  for (int i = 0; i <= m_config.frameCount; ++i) {
    qreal progress = i / static_cast<qreal>(m_config.frameCount);
    int offset = m_config.shakeDistance * std::sin(progress * 2 * M_PI) *
                 std::pow(m_config.dampingRatio, i);
    keyFrames.append({progress, m_originalPos + QPoint(offset, 0)});
  }
}

void ShakeEffect::generateVerticalShake(
    QList<QPair<qreal, QPoint>> &keyFrames) {
  for (int i = 0; i <= m_config.frameCount; ++i) {
    qreal progress = i / static_cast<qreal>(m_config.frameCount);
    int offset = m_config.shakeDistance * std::sin(progress * 2 * M_PI) *
                 std::pow(m_config.dampingRatio, i);
    keyFrames.append({progress, m_originalPos + QPoint(0, offset)});
  }
}

void ShakeEffect::generateCircularShake(
    QList<QPair<qreal, QPoint>> &keyFrames) {
  for (int i = 0; i <= m_config.frameCount; ++i) {
    qreal progress = i / static_cast<qreal>(m_config.frameCount);
    int x = m_config.shakeDistance * std::cos(progress * 2 * M_PI) *
            std::pow(m_config.dampingRatio, i);
    int y = m_config.shakeDistance * std::sin(progress * 2 * M_PI) *
            std::pow(m_config.dampingRatio, i);
    keyFrames.append({progress, m_originalPos + QPoint(x, y)});
  }
}

void ShakeEffect::generateRandomShake(QList<QPair<qreal, QPoint>> &keyFrames) {
  QRandomGenerator rand(QTime::currentTime().msec());
  for (int i = 0; i <= m_config.frameCount; ++i) {
    qreal progress = i / static_cast<qreal>(m_config.frameCount);
    int x = rand.bounded(-m_config.shakeDistance, m_config.shakeDistance);
    int y = rand.bounded(-m_config.shakeDistance, m_config.shakeDistance);
    keyFrames.append({progress, m_originalPos + QPoint(x, y)});
  }
}

void ShakeEffect::generateElasticShake(QList<QPair<qreal, QPoint>> &keyFrames) {
  for (int i = 0; i <= m_config.frameCount; ++i) {
    qreal progress = i / static_cast<qreal>(m_config.frameCount);
    qreal factor = std::pow(m_config.dampingRatio, i);
    int offset = m_config.shakeDistance * factor *
                 std::sin(progress * 6 * M_PI) * (1 - progress);
    keyFrames.append({progress, m_originalPos + QPoint(offset, 0)});
  }
}

void ShakeEffect::onAnimationFinished() {
  m_isPlaying = false;
  m_isPaused = false;
  m_target->move(m_originalPos);
  emit effectCompleted();
  spdlog::info("ShakeEffect: 动画完成");
}
