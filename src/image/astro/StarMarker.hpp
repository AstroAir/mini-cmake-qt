#pragma once

#include <filesystem>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>
#include "ParallelConfig.hpp"

namespace fs = std::filesystem;

/**
 * @struct StarMarkConfig
 * @brief 星点标注的配置参数
 */
struct StarMarkConfig {
  // 标注样式
  cv::Scalar circle_color{0, 255, 0};   ///< 圆圈颜色
  cv::Scalar text_color{255, 255, 255}; ///< 文本颜色
  int circle_thickness = 1;             ///< 圆圈线条粗细
  int text_thickness = 1;               ///< 文本粗细
  double font_scale = 0.5;              ///< 字体大小缩放
  int circle_radius = 15;               ///< 标注圆圈半径
  bool show_metrics = true;             ///< 是否显示度量值
  bool show_numbers = true;             ///< 是否显示编号
  bool antialiased = true;              ///< 是否使用抗锯齿

  // 文本布局
  int text_offset_x = 5;     ///< 文本X轴偏移
  int text_offset_y = -5;    ///< 文本Y轴偏移
  double line_spacing = 1.2; ///< 文本行间距

  // 标注过滤
  double min_fwhm = 0.0;   ///< 最小FWHM阈值
  double max_fwhm = 100.0; ///< 最大FWHM阈值

  // 输出选项
  bool save_marked_image = false;    ///< 是否保存标注后的图像
  fs::path output_path;              ///< 输出文件路径
  std::string output_format = "png"; ///< 输出文件格式

  // 新增标注样式选项
  enum class MarkerStyle {
    Circle,  ///< 圆形标注
    Cross,   ///< 十字标注
    Square,  ///< 方形标注
    Diamond, ///< 菱形标注
    Combo    ///< 组合标注
  } marker_style = MarkerStyle::Circle;

  // 新增文本布局选项
  enum class TextPosition {
    TopRight,    ///< 右上
    TopLeft,     ///< 左上
    BottomRight, ///< 右下
    BottomLeft,  ///< 左下
    Above,       ///< 正上
    Below,       ///< 正下
    Adaptive     ///< 自适应位置
  } text_position = TextPosition::TopRight;

  // 新增显示选项
  bool show_quality_score = false; ///< 是否显示质量分数
  bool show_magnitude = false;     ///< 是否显示星等
  bool show_coordinates = false;   ///< 是否显示坐标
  bool show_relative_flux = false; ///< 是否显示相对流量

  // 新增标注效果
  bool draw_shadow = true;     ///< 是否绘制阴影效果
  double shadow_opacity = 0.5; ///< 阴影不透明度
  int shadow_offset = 2;       ///< 阴影偏移量
  bool draw_glow = false;      ///< 是否绘制发光效果
  int glow_radius = 5;         ///< 发光半径
  double glow_intensity = 0.3; ///< 发光强度

  // 颜色映射选项
  bool use_color_mapping = false; ///< 是否使用颜色映射
  enum class ColorMapType {
    None,
    Quality,    ///< 基于质量的颜色映射
    Magnitude,  ///< 基于星等的颜色映射
    Temperature ///< 基于温度的颜色映射
  } color_map = ColorMapType::None;

  // 标注分组
  bool group_by_magnitude = false; ///< 按星等分组
  int magnitude_groups = 5;        ///< 星等分组数量
  bool group_by_quality = false;   ///< 按质量分组
  int quality_groups = 3;          ///< 质量分组数量

  // 标签布局优化
  bool prevent_label_overlap = true; ///< 防止标签重叠
  double min_label_distance = 20.0;  ///< 最小标签间距
  bool use_leader_lines = false;     ///< 使用引导线
  double max_leader_length = 50.0;   ///< 最大引导线长度

  // 性能优化
  bool use_parallel_processing = true; ///< 是否使用并行处理
  int thread_count = 4;                ///< 并行处理线程数
  bool use_gpu_acceleration = false;   ///< 是否使用GPU加速

  // 导出选项扩展
  bool export_metadata = false;         ///< 是否导出元数据
  std::string metadata_format = "json"; ///< 元数据格式
  bool embed_metadata = false;          ///< 是否在图像中嵌入元数据
};

/**
 * @struct StarInfo
 * @brief 星点信息结构
 */
struct StarInfo {
  cv::Point position; ///< 星点位置
  double fwhm = 0.0;  ///< FWHM值
  double hfr = 0.0;   ///< HFR值
  double snr = 0.0;   ///< 信噪比
  int index = -1;     ///< 星点编号

  // 新增属性
  double quality_score = 0.0;  ///< 质量得分
  double magnitude = 0.0;      ///< 星等
  double relative_flux = 0.0;  ///< 相对流量
  double peak_intensity = 0.0; ///< 峰值强度
  std::string catalog_id;      ///< 星表标识符
  double ra = 0.0;             ///< 赤经
  double dec = 0.0;            ///< 赤纬
  double temperature = 0.0;    ///< 温度
  int group_id = -1;           ///< 分组ID
};

/**
 * @class StarMarker
 * @brief 星点标注工具类
 */
class StarMarker {
public:
  /**
   * @brief 构造函数
   * @param config 标注配置
   */
  explicit StarMarker(StarMarkConfig config = {});

  /**
   * @brief 对图像中的星点进行标注
   * @param image 输入图像
   * @param stars 星点信息列表
   * @return 标注后的图像
   */
  cv::Mat mark_stars(const cv::Mat &image,
                     const std::vector<StarInfo> &stars) const;

  /**
   * @brief 设置标注配置
   * @param config 新的配置参数
   */
  void set_config(const StarMarkConfig &config);

  /**
   * @brief 获取当前配置
   * @return 当前的配置参数
   */
  const StarMarkConfig &get_config() const;

private:
  StarMarkConfig config_;

  /**
   * @brief 检查星点是否满足过滤条件
   * @param star 星点信息
   * @return 是否应该标注该星点
   */
  bool should_mark_star(const StarInfo &star) const;

  /**
   * @brief 为单个星点添加标注
   * @param image 图像
   * @param star 星点信息
   */
  void mark_single_star(cv::Mat &image, const StarInfo &star) const;

  /**
   * @brief 生成星点标注文本
   * @param star 星点信息
   * @return 标注文本
   */
  std::string generate_mark_text(const StarInfo &star) const;

  /**
   * @brief 保存标注后的图像
   * @param image 标注后的图像
   */
  void save_marked_image(const cv::Mat &image) const;

  // 新增私有方法
  cv::Scalar get_color_from_mapping(const StarInfo &star) const;
  void draw_marker(cv::Mat &image, const StarInfo &star) const;
  void draw_glow_effect(cv::Mat &image, const StarInfo &star) const;
  void optimize_label_positions(cv::Mat &image,
                                const std::vector<StarInfo> &stars) const;
  cv::Point calculate_optimal_text_position(const cv::Mat &image,
                                            const StarInfo &star,
                                            const cv::Size &text_size) const;
  void export_metadata(const std::vector<StarInfo> &stars) const;
  void draw_leader_line(cv::Mat &image, const cv::Point &start,
                        const cv::Point &end) const;

  /**
   * @brief 检查点是否在图像边界内
   * @param point 待检查的点
   * @param size 图像尺寸
   * @return 是否在边界内
   */
  bool check_bounds(const cv::Point &point, const cv::Size &size) const;

  /**
   * @brief 基于质量分数生成颜色
   * @param quality_score 质量分数 (0-1)
   * @return BGR颜色值
   */
  cv::Scalar color_from_quality(double quality_score) const;

  /**
   * @brief 基于星等生成颜色
   * @param magnitude 星等值
   * @return BGR颜色值
   */
  cv::Scalar color_from_magnitude(double magnitude) const;

  /**
   * @brief 基于温度生成颜色
   * @param temperature 温度值(K)
   * @return BGR颜色值
   */
  cv::Scalar color_from_temperature(double temperature) const;

  #ifdef USE_CUDA
  /**
   * @brief CUDA加速的星点标注
   */
  cv::Mat mark_stars_cuda(const cv::Mat& image,
                         const std::vector<StarInfo>& stars) const;
  
  /**
   * @brief 在GPU上处理单个星点
   */
  void process_star_cuda(cv::Mat& image, 
                        const StarInfo& star,
                        cudaStream_t stream) const;

  /**
   * @brief 检查CUDA设备状态
   */
  bool check_cuda_device() const;
  #endif

  /**
   * @brief 初始化日志系统
   */
  void init_logger() const;

  /**
   * @brief 记录性能指标
   */
  void log_performance_metrics(const std::string& operation,
                             double duration) const;
};
