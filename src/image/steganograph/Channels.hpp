#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Class for converting strings to binary and vice versa.
 */
class BinaryConverter {
public:
  /**
   * @brief Converts a string message to a vector of bits.
   * @param message The input string message.
   * @param addTerminator Whether to add a terminator (default is true).
   * @return A vector of bits representing the message.
   */
  static std::vector<bool> stringToBits(const std::string &message,
                                        bool addTerminator = true);

  /**
   * @brief Converts a vector of bits to a string message.
   * @param bits The input vector of bits.
   * @return The decoded string message.
   */
  static std::string bitsToString(const std::vector<bool> &bits);
};

/**
 * @brief Configuration structure for channel steganography.
 */
struct ChannelConfig {
  bool useBlue = true;    ///< Use the blue channel
  bool useGreen = true;   ///< Use the green channel
  bool useRed = true;     ///< Use the red channel
  bool useAlpha = true;   ///< Use the alpha channel
  int bitsPerChannel = 1; ///< Number of bits to use per channel
  double scrambleKey = 0; ///< Key for scrambling the channels

  enum class EmbedMode {
    LSB,            // 最低有效位
    RANDOM_LSB,     // 随机LSB
    ADAPTIVE_LSB    // 自适应LSB
  };
  
  enum class CompressionMode {
    NONE,
    HUFFMAN,
    LZW
  };
  
  EmbedMode embedMode = EmbedMode::LSB;
  CompressionMode compression = CompressionMode::NONE;
  bool useEncryption = false;
  std::string encryptionKey = "";
  double qualityThreshold = 0.8;  // 图像质量阈值
  bool preserveEdges = true;      // 保护边缘区域
};

/**
 * @brief Structure for analyzing the quality of a channel.
 */
struct ChannelQuality {
  double entropy;     ///< Information entropy
  double snr;         ///< Signal-to-noise ratio
  double variance;    ///< Variance
  double correlation; ///< Correlation between adjacent pixels
};

/**
 * @brief Namespace for steganography functions.
 */
namespace steganograph {

/**
 * @brief Hides a message in the alpha channel of an image.
 * @param image The input image (must have an alpha channel).
 * @param message The message to hide.
 */
void alpha_channel_hide(cv::Mat &image, const std::string &message);

/**
 * @brief Extracts a hidden message from the alpha channel of an image.
 * @param image The input image (must have an alpha channel).
 * @return The extracted message.
 */
std::string alpha_channel_extract(const cv::Mat &image);

/**
 * @brief Analyzes the channels of an image and displays them.
 * @param image The input image.
 */
void analyze_channels(const cv::Mat &image);

/**
 * @brief Calculates the capacity of an image for hiding a message.
 * @param image The input image.
 * @param config The channel configuration.
 * @return The maximum capacity in bytes.
 */
size_t calculate_capacity(const cv::Mat &image, const ChannelConfig &config);

/**
 * @brief Hides a message in multiple channels of an image.
 * @param image The input image.
 * @param message The message to hide.
 * @param config The channel configuration.
 */
void multi_channel_hide(cv::Mat &image, const std::string &message,
                        const ChannelConfig &config);

/**
 * @brief Extracts a hidden message from multiple channels of an image.
 * @param image The input image.
 * @param messageLength The length of the hidden message.
 * @param config The channel configuration.
 * @return The extracted message.
 */
std::string multi_channel_extract(const cv::Mat &image, size_t messageLength,
                                  const ChannelConfig &config);

/**
 * @brief Analyzes the quality of a single channel.
 * @param channel The input channel.
 * @return The quality metrics of the channel.
 */
ChannelQuality analyze_channel_quality(const cv::Mat &channel);

/**
 * @brief Analyzes the quality of all channels in an image.
 * @param image The input image.
 * @return A vector of quality metrics for each channel.
 */
std::vector<ChannelQuality> analyze_all_channels_quality(const cv::Mat &image);

/**
 * @brief Exception class for channel-related errors.
 */
class ChannelException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief 预处理图像以提高隐写效果
 */
void preprocess_image(cv::Mat &image, const ChannelConfig &config);

/**
 * @brief 使用自适应LSB算法隐写
 */
void adaptive_lsb_hide(cv::Mat &image, const std::string &message,
                      const ChannelConfig &config);

/**
 * @brief 使用自适应LSB算法提取
 */
std::string adaptive_lsb_extract(const cv::Mat &image,
                               const ChannelConfig &config);

/**
 * @brief 评估隐写后的图像质量
 */
double evaluate_image_quality(const cv::Mat &original,
                            const cv::Mat &modified);

/**
 * @brief 检测图像是否包含隐写内容
 */
bool detect_steganography(const cv::Mat &image,
                        double *confidence = nullptr);

} // namespace steganograph

/**
 * @brief Inline function to get the maximum message length that can be hidden
 * in an image.
 * @param image The input image.
 * @param config The channel configuration.
 * @return The maximum message length in bytes.
 */
inline size_t get_max_message_length(const cv::Mat &image,
                                     const ChannelConfig &config) {
  return steganograph::calculate_capacity(image, config);
}