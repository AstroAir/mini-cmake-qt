#ifndef IMAGEPREVIEWDIALOG_H
#define IMAGEPREVIEWDIALOG_H

#include "image/MetaData.hpp"
#include "image/StarDetector.hpp"
#include "HistogramDialog.hpp"
#include "image/Convolve.hpp"
#include "image/Exif.hpp"
#include "image/Denoise.hpp"
#include "image/Filter.hpp"

#include <QDialog>
#include <QFuture>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSlider>
#include <QToolBar>
#include <QtConcurrent>
#include <QCache>
#include <QThreadPool>


class ImagePreviewDialog : public QDialog {
  Q_OBJECT

public:
  explicit ImagePreviewDialog(QWidget *parent = nullptr);
  void setImageList(const QVector<QString> &images, int currentIndex);
  void showImage(const QString &path);
  void updateImagePreview(const QString &imagePath);
  void updateImagePreview(QLabel *label, const QString &imagePath);
  void createThumbnailGrid();

public slots:
  void updateMetadata(const ImageMetadata& metadata);
  void applyConvolution();
  void applyDeconvolution();
  void showExifInfo();

protected:
  void keyPressEvent(QKeyEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;

private slots:
  void zoomIn();
  void zoomOut();
  void rotateLeft();
  void rotateRight();
  void toggleFullscreen();
  void saveImage();
  void showNext();
  void showPrevious();
  void copyToClipboard();
  void updateZoomLabel();
  void detectStars();
  void showMetadataDialog();
  void startStarDetection();
  void onStarDetectionFinished();
  void toggleStarAnnotation();
  void updateStarAnnotation();
  void applyAutoStretch();
  void applyHistogramEqualization();
  void batchProcessImages();
  void saveImageAs();
  void saveAsFITS();
  void toggleAutoDetection();
  void exportStarData();
  void showHistogram();
  void applyCustomFilter();
  void applyBatchOperations();
  void showStatistics();
  void cropImage();
  void resizeImage();
  void adjustColors();
  void applyWatershed();
  void detectEdges();

private:
  QScrollArea *scrollArea;
  QLabel *imageLabel;
  QLabel *infoLabel;
  QLabel *countLabel;
  QLabel *zoomLabel;
  QPushButton *prevButton;
  QPushButton *nextButton;
  QToolBar *toolBar;
  QSlider *zoomSlider;

  QVector<QString> imageList;
  int currentIndex;
  qreal currentZoom;
  int currentRotation;
  bool isFullscreen;
  bool fitToWindow;
  QLabel *starDetectionLabel;
  StarDetector starDetector;
  std::vector<cv::Point> detectedStars;
  QFuture<void> starDetectionFuture;
  QProgressBar *progressBar;
  QPushButton *showMetadataButton;
  QPushButton *detectStarsButton;
  QPushButton *toggleAnnotationButton;
  bool showStarAnnotations;
  ImageProcessor imageProcessor;
  ImageMetadata currentMetadata;
  QFutureWatcher<void> starDetectionWatcher;
  QToolBar* imageProcessingToolBar;
  QAction* autoStretchAction;
  QAction* histogramEqAction;
  QAction* batchProcessAction;
  QAction* saveAsFITSAction;
  QAction* autoDetectionAction;
  QAction* exportStarDataAction;
  QAction* showHistogramAction;

  bool autoDetectionEnabled;
  cv::Mat originalImage;
  HistogramDialog* histogramDialog;

  QToolBar* createImageProcessingToolBar();
  QToolBar* createExifToolBar();
  void setupConvolutionUI();
  void processConvolution(const ConvolutionConfig& config);
  void processDeconvolution(const DeconvolutionConfig& config);

  ExifParser* exifParser;
  std::vector<ExifValue> exifData;
  QAction* showExifAction;
  QAction* applyConvolutionAction;
  QAction* applyDeconvolutionAction;
  QToolBar* convolutionToolBar;
  QToolBar* exifToolBar;

  void setupUI();
  void createToolBar();
  void updateImage();
  void updateNavigationButtons();
  void updateCountLabel();
  QString formatFileInfo(const QString &path);
  void loadImage(const QString &path);
  void scaleImage(QLabel *label, const QImage &image);
  void showStarDetectionResult();
  void setupStarDetectionUI();
  void loadFitsImage(const QString &path);
  void drawStarAnnotations(QImage &image);
  void setupImageProcessingToolBar();
  void processImage(const QString& path);
  void updateHistogram();
  double calculateImageStatistics();
  void exportDetectedStarsToCSV(const QString& path);

  QToolBar* createFilterToolBar();
  QToolBar* createBatchToolBar();
  void setupBatchProcessingUI();
  void processImageWithFilter(const QString& filterType);
  void updateImageCache();

  // 图像处理缓存
  QCache<QString, QImage> imageCache;
  // 批处理配置
  struct BatchConfig {
      bool autoStretch;
      bool denoising;
      bool edgeDetection;
      std::vector<QString> filterTypes;
  } batchConfig;

  // 图像处理线程池
  QThreadPool processingPool;

  // 图像处理相关成员
  std::unique_ptr<ImageDenoiser> denoiser;
  std::unique_ptr<ImageFilterProcessor> filterProcessor;
  std::unique_ptr<ChainImageFilterProcessor> chainProcessor;
  DenoiseParameters denoiseParams;

  // 图像处理相关方法
  void setupImageProcessingUI();
  void applyDenoising();
  void applyFilter();
  void applyChainFilters();
  void configureDenoising();
  void configureFilters();
};

#endif // IMAGEPREVIEWDIALOG_H
