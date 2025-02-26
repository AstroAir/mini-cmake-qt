#pragma once

#include <QElapsedTimer>
#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include "image/steganograph/Channels.hpp"
#include "image/steganograph/Stego.hpp"

class QLabel;
class ElaPushButton;
class QTextEdit;
class ElaComboBox;
class ElaSpinBox;
class ElaDoubleSpinBox;
class ElaCheckBox;
class ElaStatusBar;
class QGroupBox;
class ElaProgressBar;
class ElaSlider;
class QTimer;
class QVBoxLayout;
class ElaTabWidget;
class ElaLineEdit;

class StegoWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit StegoWindow(QWidget *parent = nullptr);
  double compareTextSimilarity(const std::string& text1, const std::string& text2);

signals:
  void onImageLoaded(); // 添加这个信号声明

protected:
  void resizeEvent(QResizeEvent *event) override;

private slots:
  void loadImage();
  void saveImage();
  void embedMessage();
  void extractMessage();
  void updatePreview();
  void updateChannelPreview(int channel);
  void updateBitPlanePreview(int plane);
  void switchDisplayMode(int mode);
  void updateStatusInfo();
  void embedWatermark();
  void extractWatermark();
  void compressMSB();
  void toggleMSBView();
  void updateProgressBar(int value);
  void updateStatusMessage(const QString &message);
  void onProcessingStarted();
  void onProcessingFinished();
  void updateImageQuality();
  void showOperationResult(bool success, const QString &message);
  void previewOriginalImage();
  void previewProcessedImage();
  void updatePreviewWithSize(const QSize &size);
  void setupControlPanel(QVBoxLayout *layout);
  void setupTabbedPanel();

private:
  // UI组件
  ElaTabWidget *tabWidget;
  QLabel *imagePreviewLabel;
  QLabel *histogramLabel;
  ElaStatusBar *statusBar;

  // 显示模式组件
  QGroupBox *displayGroup;
  ElaComboBox *displayModeBox;
  ElaComboBox *channelBox;
  ElaSpinBox *bitPlaneBox;

  // 图像信息组件
  QGroupBox *imageInfoGroup;
  QLabel *dimensionsLabel;
  QLabel *formatLabel;
  QLabel *capacityLabel;
  ElaProgressBar *usageBar;

  // LSB面板组件
  QWidget *createLSBPanel();
  QLabel *lsbImageLabel;
  QTextEdit *lsbMessageEdit;
  ElaPushButton *lsbEmbedButton;
  ElaPushButton *lsbExtractButton;
  ElaSpinBox *lsbBitDepthBox;
  ElaCheckBox *lsbScrambleBox;
  ElaDoubleSpinBox *lsbScrambleKeyBox;

  // 傅里叶变换面板组件
  QWidget *createFourierPanel();
  QLabel *fourierImageLabel;
  QTextEdit *fourierMessageEdit;
  ElaDoubleSpinBox *alphaSpinBox;
  ElaPushButton *fourierEmbedButton;
  ElaPushButton *fourierExtractButton;
  ElaComboBox *fourierBlockSizeBox;
  ElaCheckBox *fourierAdaptiveBox;
  ElaSlider *fourierStrengthSlider;

  // 多通道面板组件
  QWidget *createChannelPanel();
  QLabel *channelImageLabel;
  QTextEdit *channelMessageEdit;
  ElaCheckBox *blueChannelBox;
  ElaCheckBox *greenChannelBox;
  ElaCheckBox *redChannelBox;
  ElaCheckBox *alphaChannelBox;
  ElaSpinBox *bitsPerChannelBox;
  ElaDoubleSpinBox *scrambleKeyBox;
  ElaPushButton *channelEmbedButton;
  ElaPushButton *channelExtractButton;
  QGroupBox *qualityGroup;
  std::vector<ElaProgressBar *> channelQualityBars;
  ElaCheckBox *autoChannelBox;

  // 水印面板组件
  QWidget *createWatermarkPanel();
  QLabel *watermarkImageLabel;
  ElaPushButton *loadWatermarkButton;
  ElaPushButton *watermarkEmbedButton;
  ElaPushButton *watermarkExtractButton;
  ElaDoubleSpinBox *watermarkAlphaBox;
  ElaComboBox *watermarkChannelBox;
  ElaSpinBox *watermarkSizeBox;

  // MSB面板组件
  QWidget *createMSBPanel();
  QLabel *msbImageLabel;
  ElaSpinBox *msbKeepBitsBox;
  ElaPushButton *msbCompressButton;
  ElaPushButton *msbToggleButton;
  ElaProgressBar *msbQualityBar;
  ElaCheckBox *msbPreviewBox;

  // 增加新的质量监测组件
  QGroupBox *createQualityMonitorPanel();
  QLabel *psnrLabel;
  QLabel *ssimLabel;
  ElaProgressBar *qualityBar;
  QLabel *detectionLabel;

  // 增加通道质量分析组件
  void setupChannelQualityIndicators();
  std::vector<QLabel *> channelQualityLabels;

  // 增加处理控制组件
  void setupProcessingControls();
  ElaCheckBox *adaptiveProcessingBox;
  ElaCheckBox *preserveEdgesBox;
  ElaDoubleSpinBox *qualityThresholdBox;

  // 增加预处理选项
  void setupPreprocessingOptions();
  ElaComboBox *compressionModeBox;
  ElaCheckBox *encryptionBox;
  ElaLineEdit *encryptionKeyEdit;

  // 新增统计分析组件
  QGroupBox *createAnalysisPanel();
  QLabel *histogramMatrixLabel;    // 直方图矩阵显示
  QLabel *correlationMatrixLabel;  // 相关性矩阵显示
  ElaComboBox *analysisMethodBox;  // 分析方法选择
  
  // 新增隐写分析设置
  struct StegAnalysisConfig {
    bool useDeepLearning = false;  // 是否使用深度学习
    bool useHistogramAnalysis = true;  // 是否使用直方图分析
    bool useStatisticalAnalysis = true; // 是否使用统计分析
    double detectionThreshold = 0.8;    // 检测阈值
  } stegAnalysisConfig;

  // 新增隐写攻击模拟组件
  QGroupBox *createAttackSimulationPanel();
  ElaComboBox *attackTypeBox;     // 攻击类型选择
  ElaDoubleSpinBox *attackIntensityBox; // 攻击强度
  ElaPushButton *simulateAttackButton;  // 执行攻击按钮
  
  // 新增质量评估指标
  struct QualityMetrics {
    double psnr;        // 峰值信噪比
    double ssim;        // 结构相似度
    double entropy;     // 信息熵
    double correlation; // 相关性
    double robustness; // 鲁棒性得分
  } currentMetrics;

  // 通用控件
  ElaPushButton *loadImageButton;
  ElaPushButton *saveImageButton;
  ElaComboBox *previewModeBox;

  // 数据成员
  cv::Mat originalImage;
  cv::Mat processedImage;
  cv::Mat previewImage;
  cv::Mat watermarkImage;
  cv::Mat compressedImage;
  std::vector<ChannelQuality> channelQualities;
  int currentChannel;
  int currentBitPlane;
  int displayMode;
  bool msbPreviewMode;

  // 状态指示组件
  ElaProgressBar *operationProgress;
  QLabel *statusLabel;
  QLabel *qualityIndicator;
  ElaPushButton *cancelButton;
  QTimer *previewTimer;

  // 增强的预览控件
  QGroupBox *previewGroup;
  ElaPushButton *originalPreviewButton;
  ElaPushButton *processedPreviewButton;
  ElaCheckBox *autoPreviewBox;
  ElaSlider *previewSizeSlider;

  // 状态数据
  bool isProcessing;
  double currentQuality;
  QString currentOperation;
  QElapsedTimer processingTimer;

  // 辅助函数
  void setupUI();
  void createMenus();
  void updateImage(const cv::Mat &img);
  QImage matToQImage(const cv::Mat &mat);
  cv::Mat QImageToMat(const QImage &img);
  void showError(const QString &message);
  void showSuccess(const QString &message);
  void setupDisplayControls();
  void setupImageInfo();
  void setupQualityIndicators();
  void setupStatusBar();
  void updatePreviewImage(const cv::Mat &img);
  void updateHistogram(const cv::Mat &img);
  void calculateChannelQualities();
  void suggestOptimalSettings();
  void showProgress(const QString &operation, int progress);
  void loadWatermarkImage();
  void setupWatermarkControls();
  void setupMSBControls();
  void updateWatermarkPreview();
  void updateMSBPreview();
  void setupStatusIndicators();
  void setupPreviewControls();
  void updateQualityIndicator(double quality);
  void startOperation(const QString &operation);
  void finishOperation(bool success);
  QString formatProcessingTime() const;
  void enableControls(bool enable);
  void setupImageComparison();
  bool checkImageValidity();

  // 辅助功能
  void updateQualityMonitor(const cv::Mat &original, const cv::Mat &processed);
  void updateChannelAnalysis();
  void analyzeSteganography();
  void applyPreprocessing();
  void updateCapacityLabel();

  // 新增的处理配置
  ChannelConfig channelConfig;
  StegoConfig stegoConfig;

  // 状态追踪
  bool isPreprocessingEnabled;
  bool isAdaptiveEnabled;
  double currentQualityScore;

  // 新增处理函数
  void analyzeStegDetection();      // 隐写检测分析
  void simulateAttack();           // 模拟攻击
  void updateQualityMetrics();     // 更新质量指标
  void generateHistogramMatrix();   // 生成直方图矩阵
  void generateCorrelationMatrix(); // 生成相关性矩阵
  void optimizeEmbeddingStrategy(); // 优化嵌入策略
  void adaptChannelSelection();     // 自适应通道选择
  
  // 增强的质量监控
  void monitorEmbeddingQuality();   // 实时监控嵌入质量
  void suggestOptimalParameters();   // 推荐最优参数
  void validateEmbeddingCapacity(); // 验证嵌入容量
};
