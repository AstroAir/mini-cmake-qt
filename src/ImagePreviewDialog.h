#ifndef IMAGEPREVIEWDIALOG_H
#define IMAGEPREVIEWDIALOG_H

#include "HistogramDialog.hpp"
#include "image/Convolve.hpp"
#include "image/Crop.h"
#include "image/Denoise.hpp"
#include "image/Exif.hpp"
#include "image/Filter.hpp"
#include "image/MetaData.hpp"
#include "image/astro/StarDetector.hpp"

#include <QCache>
#include <QDialog>
#include <QFuture>
#include <QLabel>
#include <QMovie>
#include <QThreadPool>
#include <QtConcurrent>

class ElaStatusBar;
class ElaPushButton;
class ElaSlider;
class ElaProgressBar;
class ElaToolBar;
class ElaScrollArea;

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
  void updateMetadata(const ImageMetadata &metadata);
  void applyConvolution();
  void applyDeconvolution();
  void showExifInfo();
  void cropImage();
  void flipHorizontal();
  void flipVertical();
  void resetImage();

protected:
  void keyPressEvent(QKeyEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

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
  void applyBatchOperations();
  void showStatistics();
  void showContextMenu(const QPoint &pos);

private:
  ElaScrollArea *scrollArea;
  QLabel *imageLabel;
  QLabel *infoLabel;
  QLabel *countLabel;
  QLabel *zoomLabel;
  ElaPushButton *prevButton;
  ElaPushButton *nextButton;
  ElaToolBar *toolBar;
  ElaSlider *zoomSlider;
  ElaStatusBar *statusBar;

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
  ElaProgressBar *progressBar;
  ElaPushButton *showMetadataButton;
  ElaPushButton *detectStarsButton;
  ElaPushButton *toggleAnnotationButton;
  bool showStarAnnotations;
  ImageProcessor imageProcessor;
  ImageMetadata currentMetadata;
  QFutureWatcher<void> starDetectionWatcher;
  ElaToolBar *imageProcessingToolBar;
  QAction *autoStretchAction;
  QAction *histogramEqAction;
  QAction *batchProcessAction;
  QAction *saveAsFITSAction;
  QAction *autoDetectionAction;
  QAction *exportStarDataAction;
  QAction *showHistogramAction;

  bool autoDetectionEnabled;
  cv::Mat originalImage;
  HistogramDialog *histogramDialog;

  ElaToolBar *createImageProcessingToolBar();
  ElaToolBar *createExifToolBar();
  void setupConvolutionUI();
  void processConvolution(const ConvolutionConfig &config);
  void processDeconvolution(const DeconvolutionConfig &config);
  void initLoadingAnimation();
  void startLoading();
  void stopLoading();

  ExifParser *exifParser;
  std::vector<ExifValue> exifData;
  QAction *showExifAction;
  QAction *applyConvolutionAction;
  QAction *applyDeconvolutionAction;
  ElaToolBar *convolutionToolBar;
  ElaToolBar *exifToolBar;
  QMovie *loadingAnimation;

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
  void processImage(const QString &path);
  void updateHistogram();
  double calculateImageStatistics();
  void exportDetectedStarsToCSV(const QString &path);

  ElaToolBar *createFilterToolBar();
  ElaToolBar *createBatchToolBar();
  void setupBatchProcessingUI();
  void processImageWithFilter(const QString &filterType);
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

  ImageCropper imageCropper;

  QPointF cropStartPoint;
  QPointF cropEndPoint;
  bool isCropping = false;
  QRubberBand *rubberBand = nullptr;
  CropStrategy currentCropStrategy;

  void setupCropDialog();
  void startCrop(const QPoint &pos);
  void updateCrop(const QPoint &pos);
  void endCrop(const QPoint &pos);
  void applyCrop(const CropStrategy &strategy);
};

#endif // IMAGEPREVIEWDIALOG_H
