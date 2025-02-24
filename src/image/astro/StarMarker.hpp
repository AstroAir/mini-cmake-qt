#pragma once

#include <filesystem>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>


namespace fs = std::filesystem;

/**
 * @struct StarMarkConfig
 * @brief Configuration structure for star marking.
 */
struct StarMarkConfig {
  // Marking style
  cv::Scalar circle_color{0, 255, 0};   ///< Color of the circle.
  cv::Scalar text_color{255, 255, 255}; ///< Color of the text.
  int circle_thickness = 1;             ///< Thickness of the circle line.
  int text_thickness = 1;               ///< Thickness of the text.
  double font_scale = 0.5;              ///< Scale of the font.
  int circle_radius = 15;               ///< Radius of the marking circle.
  bool show_metrics = true;             ///< Whether to show metrics.
  bool show_numbers = true;             ///< Whether to show numbers.
  bool antialiased = true;              ///< Whether to use antialiasing.

  // Text layout
  int text_offset_x = 5;     ///< X-axis offset of the text.
  int text_offset_y = -5;    ///< Y-axis offset of the text.
  double line_spacing = 1.2; ///< Line spacing of the text.

  // Marking filter
  double min_fwhm = 0.0;   ///< Minimum FWHM threshold.
  double max_fwhm = 100.0; ///< Maximum FWHM threshold.

  // Output options
  bool save_marked_image = false;    ///< Whether to save the marked image.
  fs::path output_path;              ///< Output file path.
  std::string output_format = "png"; ///< Output file format.

  // Additional marking style options
  enum class MarkerStyle {
    Circle,  ///< Circle marking.
    Cross,   ///< Cross marking.
    Square,  ///< Square marking.
    Diamond, ///< Diamond marking.
    Combo    ///< Combo marking.
  } marker_style = MarkerStyle::Circle;

  // Additional text layout options
  enum class TextPosition {
    TopRight,    ///< Top right.
    TopLeft,     ///< Top left.
    BottomRight, ///< Bottom right.
    BottomLeft,  ///< Bottom left.
    Above,       ///< Above.
    Below,       ///< Below.
    Adaptive     ///< Adaptive position.
  } text_position = TextPosition::TopRight;

  // Additional display options
  bool show_quality_score = false; ///< Whether to show quality score.
  bool show_magnitude = false;     ///< Whether to show magnitude.
  bool show_coordinates = false;   ///< Whether to show coordinates.
  bool show_relative_flux = false; ///< Whether to show relative flux.

  // Additional marking effects
  bool draw_shadow = true;     ///< Whether to draw shadow effect.
  double shadow_opacity = 0.5; ///< Opacity of the shadow.
  int shadow_offset = 2;       ///< Offset of the shadow.
  bool draw_glow = false;      ///< Whether to draw glow effect.
  int glow_radius = 5;         ///< Radius of the glow.
  double glow_intensity = 0.3; ///< Intensity of the glow.

  // Color mapping options
  bool use_color_mapping = false; ///< Whether to use color mapping.
  enum class ColorMapType {
    None,
    Quality,    ///< Color mapping based on quality.
    Magnitude,  ///< Color mapping based on magnitude.
    Temperature ///< Color mapping based on temperature.
  } color_map = ColorMapType::None;

  // Marking grouping
  bool group_by_magnitude = false; ///< Group by magnitude.
  int magnitude_groups = 5;        ///< Number of magnitude groups.
  bool group_by_quality = false;   ///< Group by quality.
  int quality_groups = 3;          ///< Number of quality groups.

  // Label layout optimization
  bool prevent_label_overlap = true; ///< Prevent label overlap.
  double min_label_distance = 20.0;  ///< Minimum label distance.
  bool use_leader_lines = false;     ///< Use leader lines.
  double max_leader_length = 50.0;   ///< Maximum leader line length.

  // Performance optimization
  bool use_parallel_processing = true; ///< Use parallel processing.
  int thread_count = 4; ///< Number of threads for parallel processing.
  bool use_gpu_acceleration = false; ///< Use GPU acceleration.

  // Export options extension
  bool export_metadata = false;         ///< Whether to export metadata.
  std::string metadata_format = "json"; ///< Metadata format.
  bool embed_metadata = false; ///< Whether to embed metadata in the image.
};

/**
 * @struct StarInfo
 * @brief Structure representing star information.
 */
struct StarInfo {
  cv::Point position; ///< Position of the star.
  double fwhm = 0.0;  ///< FWHM value.
  double hfr = 0.0;   ///< HFR value.
  double snr = 0.0;   ///< Signal-to-noise ratio.
  int index = -1;     ///< Index of the star.

  // Additional attributes
  double quality_score = 0.0;  ///< Quality score.
  double magnitude = 0.0;      ///< Magnitude.
  double relative_flux = 0.0;  ///< Relative flux.
  double peak_intensity = 0.0; ///< Peak intensity.
  std::string catalog_id;      ///< Catalog identifier.
  double ra = 0.0;             ///< Right ascension.
  double dec = 0.0;            ///< Declination.
  double temperature = 0.0;    ///< Temperature.
  int group_id = -1;           ///< Group ID.
};

/**
 * @class StarMarker
 * @brief Class for marking stars in images.
 */
class StarMarker {
public:
  /**
   * @brief Constructor.
   * @param config Configuration for star marking.
   */
  explicit StarMarker(StarMarkConfig config = {});

  /**
   * @brief Marks stars in the image.
   * @param image Input image.
   * @param stars List of star information.
   * @return Image with marked stars.
   */
  cv::Mat mark_stars(const cv::Mat &image,
                     const std::vector<StarInfo> &stars) const;

  /**
   * @brief Sets the marking configuration.
   * @param config New configuration parameters.
   */
  void set_config(const StarMarkConfig &config);

  /**
   * @brief Gets the current configuration.
   * @return Current configuration parameters.
   */
  const StarMarkConfig &get_config() const;

private:
  StarMarkConfig config_; ///< Configuration for star marking.

  /**
   * @brief Checks if a star meets the filtering criteria.
   * @param star Star information.
   * @return Whether the star should be marked.
   */
  bool should_mark_star(const StarInfo &star) const;

  /**
   * @brief Adds a mark for a single star.
   * @param image Image.
   * @param star Star information.
   */
  void mark_single_star(cv::Mat &image, const StarInfo &star) const;

  /**
   * @brief Generates the marking text for a star.
   * @param star Star information.
   * @return Marking text.
   */
  std::string generate_mark_text(const StarInfo &star) const;

  /**
   * @brief Saves the marked image.
   * @param image Marked image.
   */
  void save_marked_image(const cv::Mat &image) const;

  // Additional private methods
  /**
   * @brief Gets the color from the mapping based on star information.
   * @param star Star information.
   * @return Color.
   */
  cv::Scalar get_color_from_mapping(const StarInfo &star) const;

  /**
   * @brief Draws the marker for a star.
   * @param image Image.
   * @param star Star information.
   */
  void draw_marker(cv::Mat &image, const StarInfo &star) const;

  /**
   * @brief Draws the glow effect for a star.
   * @param image Image.
   * @param star Star information.
   */
  void draw_glow_effect(cv::Mat &image, const StarInfo &star) const;

  /**
   * @brief Optimizes the positions of the labels.
   * @param image Image.
   * @param stars List of star information.
   */
  void optimize_label_positions(cv::Mat &image,
                                const std::vector<StarInfo> &stars) const;

  /**
   * @brief Calculates the optimal text position for a star.
   * @param image Image.
   * @param star Star information.
   * @param text_size Size of the text.
   * @return Optimal text position.
   */
  cv::Point calculate_optimal_text_position(const cv::Mat &image,
                                            const StarInfo &star,
                                            const cv::Size &text_size) const;

  /**
   * @brief Exports the metadata of the stars.
   * @param stars List of star information.
   */
  void export_metadata(const std::vector<StarInfo> &stars) const;

  /**
   * @brief Draws a leader line from the start point to the end point.
   * @param image Image.
   * @param start Start point.
   * @param end End point.
   */
  void draw_leader_line(cv::Mat &image, const cv::Point &start,
                        const cv::Point &end) const;

  /**
   * @brief Checks if a point is within the image boundaries.
   * @param point Point to check.
   * @param size Image size.
   * @return Whether the point is within the boundaries.
   */
  bool check_bounds(const cv::Point &point, const cv::Size &size) const;

  /**
   * @brief Generates a color based on the quality score.
   * @param quality_score Quality score (0-1).
   * @return BGR color value.
   */
  cv::Scalar color_from_quality(double quality_score) const;

  /**
   * @brief Generates a color based on the magnitude.
   * @param magnitude Magnitude value.
   * @return BGR color value.
   */
  cv::Scalar color_from_magnitude(double magnitude) const;

  /**
   * @brief Generates a color based on the temperature.
   * @param temperature Temperature value (K).
   * @return BGR color value.
   */
  cv::Scalar color_from_temperature(double temperature) const;

#ifdef USE_CUDA
  /**
   * @brief Marks stars using CUDA acceleration.
   * @param image Input image.
   * @param stars List of star information.
   * @return Image with marked stars.
   */
  cv::Mat mark_stars_cuda(const cv::Mat &image,
                          const std::vector<StarInfo> &stars) const;

  /**
   * @brief Processes a single star on the GPU.
   * @param image Image.
   * @param star Star information.
   * @param stream CUDA stream.
   */
  void process_star_cuda(cv::Mat &image, const StarInfo &star,
                         cudaStream_t stream) const;

  /**
   * @brief Checks the status of the CUDA device.
   * @return Whether the CUDA device is available.
   */
  bool check_cuda_device() const;
#endif

  /**
   * @brief Initializes the logging system.
   */
  void init_logger() const;

  /**
   * @brief Logs performance metrics.
   * @param operation Operation name.
   * @param duration Duration of the operation.
   */
  void log_performance_metrics(const std::string &operation,
                               double duration) const;
};