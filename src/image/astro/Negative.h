#pragma once
#include <functional>
#include <opencv2/opencv.hpp>
#include <string>

/**
 * @struct NegativeConfig
 * @brief Configuration structure for negative image processing.
 */
struct NegativeConfig {
  float intensity = 1.0f;       ///< Intensity of the negative effect.
  std::string channels = "RGB"; ///< Channels to apply the negative effect to.
  bool save_alpha = true;       ///< Whether to save the alpha channel.
  std::string output_dir = "./output"; ///< Directory to save the output images.
  cv::Rect roi = cv::Rect(0, 0, 0, 0); ///< Region of interest for processing.
  bool use_simd = true;                ///< Whether to use SIMD optimizations.
  /**
   * @brief Validates the configuration parameters.
   */
  void validate();
};

/**
 * @class NegativeProcessor
 * @brief Class for processing images to create a negative effect.
 */
class NegativeProcessor {
public:
  /**
   * @brief Constructs a NegativeProcessor with the given configuration.
   * @param cfg The configuration for negative image processing.
   */
  explicit NegativeProcessor(const NegativeConfig &cfg);

  /**
   * @brief Processes the input image to create a negative effect.
   * @param input The input image.
   * @param progress_cb Optional progress callback function.
   * @return The processed image with the negative effect.
   */
  cv::Mat process(const cv::Mat &input,
                  std::function<void(float)> progress_cb = nullptr);

private:
  NegativeConfig config_; ///< Configuration for negative image processing.
  cv::Mat lut_;           ///< Lookup table for negative effect.

  /**
   * @brief Initializes the lookup table for the negative effect.
   */
  void init_lut();

  /**
   * @brief Processes a single channel of the image.
   * @param channel The channel to process.
   */
  void process_channel(cv::Mat &channel);

  /**
   * @brief Processes a single channel of the image using SIMD optimizations.
   * @param channel The channel to process.
   */
  void process_channel_simd(cv::Mat &channel);
};

/**
 * @class NegativeApp
 * @brief Application class for running the negative image processing.
 */
class NegativeApp {
public:
  /**
   * @brief Constructs a NegativeApp.
   */
  NegativeApp();

  /**
   * @brief Runs the application.
   * @param argc The number of command-line arguments.
   * @param argv The command-line arguments.
   * @return The exit status of the application.
   */
  int run(int argc, char **argv);

private:
  NegativeConfig config_; ///< Configuration for the application.
  std::unique_ptr<NegativeProcessor>
      processor_;    ///< Processor for negative image effect.
  cv::Mat image_;    ///< The input image.
  cv::Mat negative_; ///< The processed negative image.

  /**
   * @brief Parses the command-line arguments.
   * @param argc The number of command-line arguments.
   * @param argv The command-line arguments.
   */
  void parseCommandLine(int argc, char **argv);

  /**
   * @brief Processes the input image.
   */
  void processImage();

  /**
   * @brief Saves the processed result to a file.
   * @param input_path The path of the input image file.
   */
  void saveResult(const std::string &input_path);

  /**
   * @brief Shows the help message.
   * @param parser The command-line parser.
   */
  void showHelp(const cv::CommandLineParser &parser);
};

/**
 * @brief Saves the configuration to a file.
 * @param path The file path to save the configuration.
 * @param config The configuration to save.
 */
void save_config(const std::string &path, const NegativeConfig &config);

/**
 * @brief Loads the configuration from a file.
 * @param path The file path to load the configuration from.
 * @param config The configuration to load.
 */
void load_config(const std::string &path, NegativeConfig &config);