#include <cmath>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

/**
 * @brief Channel-Temporal Attention (CTA) Unit
 *
 * CTA单元结合了通道注意力和时间注意力机制，用于视频处理：
 * 1. 通道注意力：捕获不同通道间的依赖关系
 * - 使用全局平均池化获取通道描述符
 * - 通过多层感知机学习通道间关系
 * - 应用Sigmoid激活确保权重在[0,1]范围
 *
 * 2. 时间注意力：建模时序依赖
 * - 提取时序特征
 * - 计算帧间关联度
 * - 生成时间维度的注意力权重
 */
class CTAUnit {
public:
  struct CTAConfig {
    int channels;        // 输入通道数
    int reduction_ratio; // 降维比例
    float learning_rate; // 学习率
    bool use_residual;   // 是否使用残差连接
    int temporal_window; // 时间窗口大小
  };

  explicit CTAUnit(const CTAConfig &config) : config_(config) {
    spdlog::info(
        "Initializing CTAUnit with config: channels={}, reduction_ratio={}, "
        "learning_rate={}, use_residual={}, temporal_window={}",
        config.channels, config.reduction_ratio, config.learning_rate,
        config.use_residual, config.temporal_window);
    initializeWeights();
  }

  cv::Mat forward(const std::vector<cv::Mat> &inputFrames) {
    CV_Assert(!inputFrames.empty());
    CV_Assert(inputFrames[0].channels() == config_.channels);

    spdlog::debug("CTAUnit::forward - Input frames size: {}",
                  inputFrames.size());

    // 1. 特征提取和预处理
    std::vector<cv::Mat> features = extractFeatures(inputFrames);
    spdlog::debug("CTAUnit::forward - Features extracted, size: {}",
                  features.size());

    // 2. 通道注意力计算
    cv::Mat channelAttention = computeChannelAttention(features);
    spdlog::debug("CTAUnit::forward - Channel attention computed, size: {}x{}",
                  channelAttention.rows, channelAttention.cols);

    // 3. 时间注意力计算
    cv::Mat temporalAttention = computeTemporalAttention(features);
    spdlog::debug("CTAUnit::forward - Temporal attention computed, size: {}x{}",
                  temporalAttention.rows, temporalAttention.cols);

    // 4. 注意力融合
    cv::Mat output =
        fuseAttention(features, channelAttention, temporalAttention);
    spdlog::debug("CTAUnit::forward - Attention fused, output size: {}x{}",
                  output.rows, output.cols);

    return output;
  }

private:
  CTAConfig config_;
  cv::Mat channelMLP_;  // 通道注意力MLP权重
  cv::Mat temporalMLP_; // 时间注意力MLP权重

  void initializeWeights() {
    // 初始化MLP权重，使用Xavier初始化
    float scale =
        std::sqrt(2.0f / (config_.channels +
                          config_.channels /
                              static_cast<float>(config_.reduction_ratio)));

    channelMLP_ = cv::Mat(config_.channels,
                          config_.channels / config_.reduction_ratio, CV_32F);
    temporalMLP_ =
        cv::Mat(config_.temporal_window, config_.temporal_window / 2, CV_32F);

    cv::randn(channelMLP_, 0, scale);
    cv::randn(temporalMLP_, 0, scale);

    spdlog::debug(
        "CTAUnit::initializeWeights - Channel MLP initialized, size: {}x{}",
        channelMLP_.rows, channelMLP_.cols);
    spdlog::debug(
        "CTAUnit::initializeWeights - Temporal MLP initialized, size: {}x{}",
        temporalMLP_.rows, temporalMLP_.cols);
  }

  std::vector<cv::Mat> extractFeatures(const std::vector<cv::Mat> &frames) {
    std::vector<cv::Mat> features(frames.size());

    spdlog::debug(
        "CTAUnit::extractFeatures - Extracting features from {} frames",
        frames.size());

#pragma omp parallel for
    for (int i = 0; i < frames.size(); ++i) {
      cv::Mat feat;
      frames[i].convertTo(feat, CV_32F);
      // 应用全局平均池化
      std::vector<cv::Mat> channels;
      cv::split(feat, channels);

      features[i] = cv::Mat(1, config_.channels, CV_32F);
      for (int c = 0; c < channels.size(); ++c) {
        features[i].at<float>(0, c) = cv::mean(channels[c])[0];
      }
      spdlog::trace("CTAUnit::extractFeatures - Feature extracted for frame "
                    "{}, size: {}x{}",
                    i, features[i].rows, features[i].cols);
    }
    spdlog::debug("CTAUnit::extractFeatures - All features extracted");
    return features;
  }

  cv::Mat computeChannelAttention(const std::vector<cv::Mat> &features) {
    cv::Mat channelDesc = cv::Mat::zeros(1, config_.channels, CV_32F);

    // 计算通道描述符
    for (const auto &feat : features) {
      channelDesc += feat;
    }
    channelDesc /= features.size();

    spdlog::trace("CTAUnit::computeChannelAttention - Channel descriptor "
                  "computed, size: {}x{}",
                  channelDesc.rows, channelDesc.cols);

    // 通过MLP学习通道关系
    cv::Mat attention;
    cv::gemm(channelDesc, channelMLP_, 1.0, cv::Mat(), 0.0, attention);

    // 应用ReLU
    attention = cv::max(attention, 0);

    // 恢复维度
    cv::gemm(attention, channelMLP_.t(), 1.0, cv::Mat(), 0.0, attention);

    // Sigmoid激活
    cv::exp(-attention, attention);
    attention = 1.0 / (1.0 + attention);

    spdlog::trace("CTAUnit::computeChannelAttention - Channel attention "
                  "computed, size: {}x{}",
                  attention.rows, attention.cols);
    return attention;
  }

  cv::Mat computeTemporalAttention(const std::vector<cv::Mat> &features) {
    int numFrames = features.size();
    cv::Mat temporalFeat = cv::Mat::zeros(numFrames, config_.channels, CV_32F);

    // 构建时序特征
    for (int i = 0; i < numFrames; ++i) {
      features[i].copyTo(temporalFeat.row(i));
    }

    spdlog::trace("CTAUnit::computeTemporalAttention - Temporal features "
                  "constructed, size: {}x{}",
                  temporalFeat.rows, temporalFeat.cols);

    // 计算时间注意力
    cv::Mat attention;
    cv::gemm(temporalFeat, temporalMLP_, 1.0, cv::Mat(), 0.0, attention);

    // 应用Softmax确保权重和为1
    cv::exp(attention, attention);
    cv::Mat rowSum;
    cv::reduce(attention, rowSum, 1, cv::REDUCE_SUM);
    for (int i = 0; i < attention.rows; ++i) {
      attention.row(i) /= rowSum.at<float>(i, 0) + 1e-6f;
    }

    spdlog::trace("CTAUnit::computeTemporalAttention - Temporal attention "
                  "computed, size: {}x{}",
                  attention.rows, attention.cols);
    return attention;
  }

  cv::Mat fuseAttention(const std::vector<cv::Mat> &features,
                        const cv::Mat &channelAttention,
                        const cv::Mat &temporalAttention) {
    cv::Mat output = cv::Mat::zeros(features[0].size(), CV_32F);

    spdlog::debug("CTAUnit::fuseAttention - Fusing attention");

#pragma omp parallel for collapse(2)
    for (int t = 0; t < features.size(); ++t) {
      for (int c = 0; c < config_.channels; ++c) {
        float weight = channelAttention.at<float>(0, c) *
                       temporalAttention.at<float>(t, 0);
        output += features[t].col(c) * weight;
      }
    }

    // 添加残差连接
    if (config_.use_residual) {
      output += features[features.size() / 2]; // 使用中间帧作为残差
      spdlog::trace("CTAUnit::fuseAttention - Residual connection added");
    }

    spdlog::debug("CTAUnit::fuseAttention - Attention fused");
    return output;
  }
};

// 使用示例
void testCTA() {
  CTAUnit::CTAConfig config{.channels = 3,
                            .reduction_ratio = 16,
                            .learning_rate = 0.001f,
                            .use_residual = true,
                            .temporal_window = 5};

  CTAUnit cta(config);

  // 创建测试视频帧
  std::vector<cv::Mat> frames;
  for (int i = 0; i < config.temporal_window; ++i) {
    cv::Mat frame(256, 256, CV_32FC3);
    cv::randu(frame, 0, 1);
    frames.push_back(frame);
  }

  // 处理帧
  cv::Mat result = cta.forward(frames);

  // 显示结果
  cv::imshow("CTA Result", result);
  cv::waitKey(0);
}
