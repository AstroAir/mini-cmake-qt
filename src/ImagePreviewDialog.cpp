#include "ImagePreviewDialog.h"

#include "MetadataDialog.h"
#include "image/ImageIO.hpp"

#include <QClipboard>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QImageReader>
#include <QInputDialog>
#include <QKeyEvent>
#include <QMessageBox>
#include <QPainter>
#include <QProgressDialog>
#include <QScreen>
#include <QShortcut>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

ImagePreviewDialog::ImagePreviewDialog(QWidget *parent)
    : QDialog(parent), currentZoom(1.0), currentRotation(0),
      isFullscreen(false), fitToWindow(true) { // 默认适应窗口
  histogramDialog = new HistogramDialog(this);
  setupUI();
  resize(800, 600);
  setWindowTitle("图片预览");
}

void ImagePreviewDialog::setupUI() {
  auto *mainLayout = new QVBoxLayout(this);
  mainLayout->setContentsMargins(0, 0, 0, 0);
  mainLayout->setSpacing(0);

  // 创建工具栏
  createToolBar();
  mainLayout->addWidget(toolBar);

  // 创建信息栏
  auto *infoBar = new QWidget;
  auto *infoLayout = new QHBoxLayout(infoBar);
  infoLayout->setContentsMargins(5, 5, 5, 5);

  prevButton = new QPushButton("上一张");
  nextButton = new QPushButton("下一张");
  countLabel = new QLabel;
  infoLabel = new QLabel;
  zoomLabel = new QLabel;

  infoLayout->addWidget(prevButton);
  infoLayout->addWidget(countLabel);
  infoLayout->addWidget(nextButton);
  infoLayout->addStretch();
  infoLayout->addWidget(zoomLabel);
  infoLayout->addWidget(infoLabel);

  // 在主布局中添加星点检测结果标签
  starDetectionLabel = new QLabel;
  starDetectionLabel->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
  starDetectionLabel->setStyleSheet("QLabel { color: green; }");
  infoLayout->addWidget(starDetectionLabel);

  mainLayout->addWidget(infoBar);

  // 创建图片显示区域
  scrollArea = new QScrollArea;
  imageLabel = new QLabel;
  imageLabel->setAlignment(Qt::AlignCenter);
  scrollArea->setWidget(imageLabel);
  scrollArea->setWidgetResizable(true);
  mainLayout->addWidget(scrollArea);

  // 连接信号
  connect(prevButton, &QPushButton::clicked, this,
          &ImagePreviewDialog::showPrevious);
  connect(nextButton, &QPushButton::clicked, this,
          &ImagePreviewDialog::showNext);

  // 添加快捷键
  new QShortcut(QKeySequence::ZoomIn, this, this, &ImagePreviewDialog::zoomIn);
  new QShortcut(QKeySequence::ZoomOut, this, this,
                &ImagePreviewDialog::zoomOut);
  new QShortcut(Qt::Key_Left, this, this, &ImagePreviewDialog::showPrevious);
  new QShortcut(Qt::Key_Right, this, this, &ImagePreviewDialog::showNext);
  new QShortcut(Qt::Key_F11, this, this, &ImagePreviewDialog::toggleFullscreen);

  // 添加星点检测和元信息相关按钮
  setupStarDetectionUI();
  setupImageProcessingUI();
  mainLayout->addWidget(imageProcessingToolBar);

  // 添加卷积和EXIF工具栏
  convolutionToolBar = createImageProcessingToolBar();
  exifToolBar = createExifToolBar();
  mainLayout->addWidget(convolutionToolBar);
  mainLayout->addWidget(exifToolBar);
}

void ImagePreviewDialog::createToolBar() {
  toolBar = new QToolBar;
  toolBar->setIconSize(QSize(24, 24));

  auto addToolButton = [this](const QString &text, const QIcon &icon,
                              auto slot) {
    QToolButton *btn = new QToolButton;
    btn->setIcon(icon);
    btn->setToolTip(text);
    connect(btn, &QToolButton::clicked, this, slot);
    toolBar->addWidget(btn);
  };

  addToolButton("放大", style()->standardIcon(QStyle::SP_TitleBarMaxButton),
                &ImagePreviewDialog::zoomIn);
  addToolButton("缩小", style()->standardIcon(QStyle::SP_TitleBarMinButton),
                &ImagePreviewDialog::zoomOut);
  toolBar->addSeparator();
  addToolButton("向左旋转", QIcon::fromTheme("object-rotate-left"),
                &ImagePreviewDialog::rotateLeft);
  addToolButton("向右旋转", QIcon::fromTheme("object-rotate-right"),
                &ImagePreviewDialog::rotateRight);
  toolBar->addSeparator();
  addToolButton("全屏", style()->standardIcon(QStyle::SP_TitleBarMaxButton),
                &ImagePreviewDialog::toggleFullscreen);
  addToolButton("复制", style()->standardIcon(QStyle::SP_DialogSaveButton),
                &ImagePreviewDialog::copyToClipboard);
  addToolButton("保存", style()->standardIcon(QStyle::SP_DialogSaveButton),
                &ImagePreviewDialog::saveImage);

  // 添加星点识别按钮
  addToolButton("星点识别", style()->standardIcon(QStyle::SP_DialogApplyButton),
                &ImagePreviewDialog::detectStars);

  // 添加缩放滑块
  zoomSlider = new QSlider(Qt::Horizontal);
  zoomSlider->setRange(10, 400);
  zoomSlider->setValue(100);
  zoomSlider->setFixedWidth(100);
  connect(zoomSlider, &QSlider::valueChanged, this, [this](int value) {
    currentZoom = value / 100.0;
    updateImage();
  });
  toolBar->addWidget(zoomSlider);
}

// ... 续上文 ...

void ImagePreviewDialog::setImageList(const QVector<QString> &images,
                                      int currentIdx) {
  imageList = images;
  currentIndex = currentIdx;
  if (!images.isEmpty()) {
    loadImage(images[currentIdx]);
    updateNavigationButtons();
    updateCountLabel();
  }
}

void ImagePreviewDialog::loadImage(const QString &path) {
  try {
    spdlog::info("Loading image: {}", path.toStdString());

    if (path.isEmpty()) {
      throw std::runtime_error("图片路径为空");
    }

    QFileInfo fileInfo(path);
    if (!fileInfo.exists()) {
      throw std::runtime_error("图片文件不存在");
    }

    if (!fileInfo.isReadable()) {
      throw std::runtime_error("无法读取图片文件,请检查权限");
    }

    if (path.endsWith(".fits", Qt::CaseInsensitive)) {
      loadFitsImage(path);
      return;
    }

    QImageReader reader(path);
    reader.setAutoTransform(true);

    if (!reader.canRead()) {
      throw std::runtime_error("无法识别的图片格式: " +
                               reader.errorString().toStdString());
    }

    QImage image = reader.read();
    if (image.isNull()) {
      throw std::runtime_error("读取图片失败: " +
                               reader.errorString().toStdString());
    }

    // 读取原始图像用于处理
    originalImage = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
    if (originalImage.empty()) {
      spdlog::warn("Failed to load original image for processing");
    }

    // 更新界面显示
    infoLabel->setText(formatFileInfo(path));
    imageLabel->setPixmap(QPixmap::fromImage(image));
    updateImage();

    // 更新窗口标题
    setWindowTitle(QString("图片预览 - %1").arg(fileInfo.fileName()));

    // 自动检测星点
    if (autoDetectionEnabled) {
      startStarDetection();
    }

    // 更新直方图
    updateHistogram();

    spdlog::info("Successfully loaded image: {}", path.toStdString());

  } catch (const std::exception &e) {
    spdlog::error("Error loading image: {}", e.what());
    QMessageBox::critical(this, "加载错误",
                          QString("无法加载图片：%1").arg(e.what()));

    // 清除当前显示
    imageLabel->clear();
    infoLabel->clear();
    originalImage.release();
  }
}

void ImagePreviewDialog::loadFitsImage(const QString &path) {
  try {
    cv::Mat image;
    // 使用 ImageIO 中的 FITS 读取功能
    // TODO: 实现FITS图像读取功能
    if (!image.empty()) {
      QImage qImage(image.data, image.cols, image.rows, image.step,
                    QImage::Format_Grayscale8);
      imageLabel->setPixmap(QPixmap::fromImage(qImage));
      updateImage();
    }
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "错误",
                         QString("无法加载FITS图像: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::updateImage() {
  QPixmap currentPixmap = imageLabel->pixmap(Qt::ReturnByValue);
  if (!currentPixmap.isNull()) {
    QPixmap rotatedPixmap =
        currentPixmap.transformed(QTransform().rotate(currentRotation));

    QSize size = rotatedPixmap.size() * currentZoom;
    imageLabel->setPixmap(rotatedPixmap.scaled(size, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation));
    updateZoomLabel();
  }
}

void ImagePreviewDialog::keyPressEvent(QKeyEvent *event) {
  switch (event->key()) {
  case Qt::Key_Escape:
    if (isFullscreen) {
      toggleFullscreen();
    } else {
      close();
    }
    break;
  default:
    QDialog::keyPressEvent(event);
  }
}

void ImagePreviewDialog::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() & Qt::ControlModifier) {
    if (event->angleDelta().y() > 0) {
      zoomIn();
    } else {
      zoomOut();
    }
    event->accept();
  } else {
    QDialog::wheelEvent(event);
  }
}

void ImagePreviewDialog::zoomIn() {
  currentZoom *= 1.2;
  zoomSlider->setValue(currentZoom * 100);
  updateImage();
}

void ImagePreviewDialog::zoomOut() {
  currentZoom /= 1.2;
  zoomSlider->setValue(currentZoom * 100);
  updateImage();
}

void ImagePreviewDialog::rotateLeft() {
  currentRotation -= 90;
  updateImage();
}

void ImagePreviewDialog::rotateRight() {
  currentRotation += 90;
  updateImage();
}

void ImagePreviewDialog::toggleFullscreen() {
  if (isFullscreen) {
    showNormal();
  } else {
    showFullScreen();
  }
  isFullscreen = !isFullscreen;

  // 重新调整图片大小
  if (!imageLabel->pixmap(Qt::ReturnByValue).isNull()) {
    updateImagePreview(imageList[currentIndex]);
  }
}

void ImagePreviewDialog::updateNavigationButtons() {
  prevButton->setEnabled(currentIndex > 0);
  nextButton->setEnabled(currentIndex < imageList.size() - 1);
}

void ImagePreviewDialog::updateCountLabel() {
  countLabel->setText(
      QString("%1 / %2").arg(currentIndex + 1).arg(imageList.size()));
}

void ImagePreviewDialog::updateZoomLabel() {
  zoomLabel->setText(QString("缩放: %1%").arg(int(currentZoom * 100)));
}

QString ImagePreviewDialog::formatFileInfo(const QString &path) {
  QFileInfo info(path);
  return QString("%1 (%2, %3)")
      .arg(info.fileName())
      .arg(info.size() / 1024.0, 0, 'f', 1)
      .arg(info.suffix().toUpper());
}

void ImagePreviewDialog::copyToClipboard() {
  QPixmap currentPixmap = imageLabel->pixmap(Qt::ReturnByValue);
  if (!currentPixmap.isNull()) {
    QGuiApplication::clipboard()->setPixmap(currentPixmap);
  }
}

void ImagePreviewDialog::saveImage() {
  QPixmap currentPixmap = imageLabel->pixmap(Qt::ReturnByValue);
  if (currentPixmap.isNull())
    return;

  QString fileName =
      QFileDialog::getSaveFileName(this, "保存图片", imageList[currentIndex],
                                   "图片文件 (*.png *.jpg *.bmp)");

  if (!fileName.isEmpty()) {
    currentPixmap.save(fileName);
  }
}

void ImagePreviewDialog::saveImageAs() {
  QPixmap currentPixmap = imageLabel->pixmap(Qt::ReturnByValue);
  if (currentPixmap.isNull())
    return;

  QString fileName = QFileDialog::getSaveFileName(
      this, "另存为", imageList[currentIndex],
      "图像文件 (*.png *.jpg *.bmp);;FITS文件 (*.fits);;所有文件 (*.*)");

  if (fileName.isEmpty())
    return;

  if (fileName.endsWith(".fits", Qt::CaseInsensitive)) {
    saveAsFITS();
  } else {
    currentPixmap.save(fileName);
  }
}

void ImagePreviewDialog::showNext() {
  if (currentIndex < imageList.size() - 1) {
    try {
      currentIndex++;
      loadImage(imageList[currentIndex]);
      updateNavigationButtons();
      updateCountLabel();
    } catch (const std::exception &e) {
      spdlog::error("Error showing next image: {}", e.what());
      QMessageBox::warning(this, "错误",
                           QString("无法显示下一张图片：%1").arg(e.what()));
    }
  }
}

void ImagePreviewDialog::showPrevious() {
  if (currentIndex > 0) {
    try {
      currentIndex--;
      loadImage(imageList[currentIndex]);
      updateNavigationButtons();
      updateCountLabel();
    } catch (const std::exception &e) {
      spdlog::error("Error showing previous image: {}", e.what());
      QMessageBox::warning(this, "错误",
                           QString("无法显示上一张图片：%1").arg(e.what()));
    }
  }
}

void ImagePreviewDialog::updateImagePreview(const QString &imagePath) {
  updateImagePreview(imageLabel, imagePath);
}

void ImagePreviewDialog::updateImagePreview(QLabel *label,
                                            const QString &imagePath) {
  QImageReader reader(imagePath);
  reader.setAutoTransform(true);
  QImage image = reader.read();

  if (!image.isNull()) {
    // 更新文件信息
    QFileInfo fileInfo(imagePath);
    infoLabel->setText(formatFileInfo(imagePath));

    // 应用旋转
    if (currentRotation != 0) {
      image = image.transformed(QTransform().rotate(currentRotation));
    }

    scaleImage(label, image);
  } else {
    label->clear();
    label->setText(tr("无法加载图片"));
  }
}

void ImagePreviewDialog::scaleImage(QLabel *label, const QImage &image) {
  QSize viewSize = scrollArea->viewport()->size();
  QSize imageSize = image.size();

  // 计算缩放尺寸
  QSize targetSize;
  if (fitToWindow) {
    targetSize = viewSize;
  } else {
    targetSize = imageSize * currentZoom;
  }

  // 保持宽高比
  targetSize = imageSize.scaled(targetSize, Qt::KeepAspectRatio);

  // 创建缩放后的图片
  QPixmap scaledPixmap = QPixmap::fromImage(image).scaled(
      targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

  label->setPixmap(scaledPixmap);

  // 更新缩放标签
  updateZoomLabel();
}

// 优化：改进星点检测算法，增加并发处理和精度计算（仅增加注释，具体实现依赖算法细节）
void ImagePreviewDialog::detectStars() {
  if (currentIndex >= 0 && currentIndex < imageList.size()) {
    QString imagePath = imageList[currentIndex];
    cv::Mat image = cv::imread(imagePath.toStdString(), cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
      QMessageBox::warning(this, "错误", "无法加载图像进行星点检测");
      return;
    }

    // ...existing preprocessing code...
    // 新增：优化算法精度，调用高性能运算库或多线程并行执行（示例伪代码）
    // cv::parallel_for(0, image.rows, [&](int i){ /* 计算每行星点检测 */ });

    try {
      // 配置星点检测参数
      StarDetectionConfig config;
      config.visualize = false; // 关闭默认可视化
      config.min_star_size = 5;
      config.min_star_brightness = 30;

      // 执行星点检测
      starDetector = StarDetector(config);
      detectedStars = starDetector.multiscale_detect_stars(image);

      // 显示结果
      showStarDetectionResult();

    } catch (const std::exception &e) {
      QMessageBox::warning(this, "错误",
                           QString("星点检测失败: %1").arg(e.what()));
    }
  }
}

void ImagePreviewDialog::showStarDetectionResult() {
  if (detectedStars.empty()) {
    starDetectionLabel->setText("未检测到星点");
    return;
  }

  // 在当前图像上标记星点
  QPixmap currentPixmap = imageLabel->pixmap(Qt::ReturnByValue);
  QImage image = currentPixmap.toImage();
  QPainter painter(&image);

  painter.setPen(QPen(Qt::red, 2));

  for (const auto &star : detectedStars) {
    // 考虑当前缩放比例调整坐标
    int x = star.x * currentZoom;
    int y = star.y * currentZoom;
    painter.drawEllipse(QPoint(x, y), 5, 5);
  }

  imageLabel->setPixmap(QPixmap::fromImage(image));
  starDetectionLabel->setText(
      QString("检测到 %1 个星点").arg(detectedStars.size()));
}

void ImagePreviewDialog::setupStarDetectionUI() {
  showMetadataButton = new QPushButton("显示元信息");
  detectStarsButton = new QPushButton("检测星点");
  toggleAnnotationButton = new QPushButton("显示/隐藏标注");
  toggleAnnotationButton->setEnabled(false);
  progressBar = new QProgressBar;
  progressBar->setVisible(false);

  toolBar->addSeparator();
  toolBar->addWidget(showMetadataButton);
  toolBar->addWidget(detectStarsButton);
  toolBar->addWidget(toggleAnnotationButton);
  toolBar->addWidget(progressBar);

  connect(showMetadataButton, &QPushButton::clicked, this,
          &ImagePreviewDialog::showMetadataDialog);
  connect(detectStarsButton, &QPushButton::clicked, this,
          &ImagePreviewDialog::startStarDetection);
  connect(toggleAnnotationButton, &QPushButton::clicked, this,
          &ImagePreviewDialog::toggleStarAnnotation);
}

void ImagePreviewDialog::startStarDetection() {
  if (starDetectionFuture.isRunning()) {
    return;
  }

  detectStarsButton->setEnabled(false);
  progressBar->setVisible(true);
  progressBar->setRange(0, 0);

  // 在新线程中执行星点检测
  starDetectionFuture = QtConcurrent::run([this]() {
    try {
      StarDetectionConfig config;
      // 配置检测参数
      config.min_star_size = 5;
      config.min_star_brightness = 30;
      config.visualize = false;

      cv::Mat image = cv::imread(imageList[currentIndex].toStdString(),
                                 cv::IMREAD_GRAYSCALE);
      starDetector = StarDetector(config);
      detectedStars = starDetector.multiscale_detect_stars(image);
    } catch (const std::exception &e) {
      QMetaObject::invokeMethod(this, [this, error = QString(e.what())]() {
        QMessageBox::warning(this, "错误", "星点检测失败: " + error);
      });
    }
  });

  starDetectionWatcher.setFuture(starDetectionFuture);
  connect(&starDetectionWatcher, &QFutureWatcher<void>::finished, this,
          &ImagePreviewDialog::onStarDetectionFinished);
}

void ImagePreviewDialog::onStarDetectionFinished() {
  progressBar->setVisible(false);
  detectStarsButton->setEnabled(true);
  toggleAnnotationButton->setEnabled(!detectedStars.empty());

  if (!detectedStars.empty()) {
    QMessageBox::information(
        this, "检测完成",
        QString("检测到 %1 个星点").arg(detectedStars.size()));
    showStarAnnotations = true;
    updateStarAnnotation();
  }
}

void ImagePreviewDialog::toggleStarAnnotation() {
  showStarAnnotations = !showStarAnnotations;
  updateStarAnnotation();
}

void ImagePreviewDialog::updateStarAnnotation() {
  QImage image = imageLabel->pixmap(Qt::ReturnByValue).toImage();
  if (showStarAnnotations) {
    drawStarAnnotations(image);
  }
  imageLabel->setPixmap(QPixmap::fromImage(image));
}

void ImagePreviewDialog::drawStarAnnotations(QImage &image) {
  QPainter painter(&image);
  painter.setPen(QPen(Qt::red, 2));

  for (const auto &star : detectedStars) {
    // 考虑缩放和旋转
    QPointF scaledPos(star.x * currentZoom, star.y * currentZoom);
    if (currentRotation != 0) {
      QTransform transform;
      transform.rotate(currentRotation);
      scaledPos = transform.map(scaledPos);
    }
    painter.drawEllipse(scaledPos, 5, 5);
  }
}

void ImagePreviewDialog::showMetadataDialog() {
  if (!currentMetadata.path.empty()) {
    MetadataDialog dialog(currentMetadata, this);
    if (dialog.exec() == QDialog::Accepted) {
      updateMetadata(dialog.getMetadata());
    }
  }
}

void ImagePreviewDialog::updateMetadata(const ImageMetadata &metadata) {
  currentMetadata = metadata;
  try {
    // 保存更新后的元数据到文件
    imageProcessor.save_metadata(currentMetadata);
    QMessageBox::information(this, "成功", "元数据已保存到文件");
  } catch (const std::exception &e) {
    QMessageBox::critical(this, "错误",
                          QString("保存元数据失败: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::setupImageProcessingUI() {
  imageProcessingToolBar = new QToolBar("图像处理", this);

  // 降噪相关按钮
  auto denoiseButton = imageProcessingToolBar->addAction("降噪");
  connect(denoiseButton, &QAction::triggered, this,
          &ImagePreviewDialog::configureDenoising);

  // 滤镜相关按钮
  auto filterButton = imageProcessingToolBar->addAction("滤镜");
  connect(filterButton, &QAction::triggered, this,
          &ImagePreviewDialog::configureFilters);

  imageProcessingToolBar->addSeparator();

  // 直方图按钮
  showHistogramAction = imageProcessingToolBar->addAction(
      "显示直方图", this, &ImagePreviewDialog::showHistogram);
}

void ImagePreviewDialog::configureDenoising() {
  QDialog dialog(this);
  dialog.setWindowTitle("降噪设置");
  auto layout = new QVBoxLayout(&dialog);

  // 创建降噪方法选择下拉框
  auto methodCombo = new QComboBox(&dialog);
  methodCombo->addItems(
      {"自动", "中值滤波", "高斯滤波", "双边滤波", "NLM", "小波"});
  layout->addWidget(new QLabel("降噪方法:"));
  layout->addWidget(methodCombo);

  // 参数设置控件
  auto strengthSlider = new QSlider(Qt::Horizontal, &dialog);
  strengthSlider->setRange(1, 100);
  strengthSlider->setValue(50);
  layout->addWidget(new QLabel("强度:"));
  layout->addWidget(strengthSlider);

  // 确定取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
  layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  if (dialog.exec() == QDialog::Accepted) {
    // 配置降噪参数
    denoiseParams.method =
        static_cast<DenoiseMethod>(methodCombo->currentIndex());
    double strength = strengthSlider->value() / 50.0;

    denoiseParams.nlm_h = 3.0 * strength;
    denoiseParams.bilateral_d = static_cast<int>(9 * strength);
    denoiseParams.wavelet_threshold = 15.0f * strength;

    applyDenoising();
  }
}

void ImagePreviewDialog::applyDenoising() {
  if (originalImage.empty())
    return;

  try {
    QProgressDialog progress("正在降噪...", "取消", 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    if (!denoiser) {
      denoiser = std::make_unique<ImageDenoiser>();
    }

    cv::Mat result = denoiser->denoise(originalImage, denoiseParams);

    QImage qImage;
    if (result.channels() == 1) {
      qImage = QImage(result.data, result.cols, result.rows, result.step,
                      QImage::Format_Grayscale8);
    } else {
      qImage = QImage(result.data, result.cols, result.rows, result.step,
                      QImage::Format_BGR888);
    }

    imageLabel->setPixmap(QPixmap::fromImage(qImage));
    updateImage();

  } catch (const std::exception &e) {
    QMessageBox::warning(this, "错误",
                         QString("降噪处理失败: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::configureFilters() {
  QDialog dialog(this);
  dialog.setWindowTitle("滤镜设置");
  auto layout = new QVBoxLayout(&dialog);

  // 滤镜选择列表
  auto filterList = new QListWidget(&dialog);
  filterList->addItems(
      {"高斯模糊", "边缘检测", "锐化", "HSV调整", "对比度亮度"});
  filterList->setSelectionMode(QAbstractItemView::ExtendedSelection);
  layout->addWidget(new QLabel("选择滤镜:"));
  layout->addWidget(filterList);

  // 确定取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
  layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  if (dialog.exec() == QDialog::Accepted) {
    // 创建选中的滤镜链
    std::vector<std::unique_ptr<IFilterStrategy>> filters;

    for (auto item : filterList->selectedItems()) {
      if (item->text() == "高斯模糊") {
        filters.push_back(std::make_unique<GaussianBlurFilter>());
      } else if (item->text() == "边缘检测") {
        filters.push_back(std::make_unique<CannyEdgeFilter>());
      } else if (item->text() == "锐化") {
        filters.push_back(std::make_unique<SharpenFilter>());
      } else if (item->text() == "HSV调整") {
        filters.push_back(std::make_unique<HSVAdjustFilter>());
      } else if (item->text() == "对比度亮度") {
        filters.push_back(std::make_unique<ContrastBrightnessFilter>());
      }
    }

    if (!filters.empty()) {
      chainProcessor =
          std::make_unique<ChainImageFilterProcessor>(std::move(filters));
      applyChainFilters();
    }
  }
}

void ImagePreviewDialog::applyChainFilters() {
  if (!chainProcessor || !imageLabel->pixmap())
    return;

  try {
    QImage currentImage = imageLabel->pixmap().toImage();
    QImage processed = chainProcessor->process(currentImage);
    imageLabel->setPixmap(QPixmap::fromImage(processed));
    updateImage();
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "错误",
                         QString("滤镜处理失败: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::setupImageProcessingToolBar() {
  imageProcessingToolBar = new QToolBar("图像处理", this);

  autoStretchAction = imageProcessingToolBar->addAction(
      "自动拉伸", this, &ImagePreviewDialog::applyAutoStretch);
  histogramEqAction = imageProcessingToolBar->addAction(
      "直方图均衡", this, &ImagePreviewDialog::applyHistogramEqualization);
  imageProcessingToolBar->addSeparator();

  batchProcessAction = imageProcessingToolBar->addAction(
      "批处理", this, &ImagePreviewDialog::batchProcessImages);
  saveAsFITSAction = imageProcessingToolBar->addAction(
      "另存为FITS", this, &ImagePreviewDialog::saveAsFITS);
  autoDetectionAction = imageProcessingToolBar->addAction(
      "自动检测", this, &ImagePreviewDialog::toggleAutoDetection);
  autoDetectionAction->setCheckable(true);

  exportStarDataAction = imageProcessingToolBar->addAction(
      "导出星点数据", this, &ImagePreviewDialog::exportStarData);
  showHistogramAction = imageProcessingToolBar->addAction(
      "显示直方图", this, &ImagePreviewDialog::showHistogram);
}

QToolBar *ImagePreviewDialog::createImageProcessingToolBar() {
  auto toolbar = new QToolBar("图像处理", this);

  applyConvolutionAction = toolbar->addAction(
      "应用卷积", this, &ImagePreviewDialog::applyConvolution);
  applyDeconvolutionAction = toolbar->addAction(
      "应用反卷积", this, &ImagePreviewDialog::applyDeconvolution);

  return toolbar;
}

QToolBar *ImagePreviewDialog::createExifToolBar() {
  auto toolbar = new QToolBar("EXIF信息", this);

  showExifAction =
      toolbar->addAction("查看EXIF", this, &ImagePreviewDialog::showExifInfo);

  return toolbar;
}

void ImagePreviewDialog::applyAutoStretch() {
  if (originalImage.empty())
    return;

  cv::Mat processed;
  double minVal, maxVal;
  cv::minMaxLoc(originalImage, &minVal, &maxVal);

  originalImage.convertTo(processed, -1, 255.0 / (maxVal - minVal),
                          -minVal * 255.0 / (maxVal - minVal));

  QImage qImage(processed.data, processed.cols, processed.rows, processed.step,
                QImage::Format_Grayscale8);
  imageLabel->setPixmap(QPixmap::fromImage(qImage));
  updateImage();
}

void ImagePreviewDialog::applyHistogramEqualization() {
  if (originalImage.empty())
    return;

  cv::Mat processed;
  cv::equalizeHist(originalImage, processed);

  QImage qImage(processed.data, processed.cols, processed.rows, processed.step,
                QImage::Format_Grayscale8);
  imageLabel->setPixmap(QPixmap::fromImage(qImage));
  updateImage();
}

void ImagePreviewDialog::batchProcessImages() {
  QStringList options;
  options << "星点检测" << "自动拉伸" << "直方图均衡" << "全部处理";

  bool ok;
  QString selected = QInputDialog::getItem(this, "批处理", "选择处理方式：",
                                           options, 0, false, &ok);

  if (!ok || selected.isEmpty())
    return;

  QProgressDialog progress("处理中...", "取消", 0, imageList.size(), this);
  progress.setWindowModality(Qt::WindowModal);

  for (int i = 0; i < imageList.size(); ++i) {
    if (progress.wasCanceled())
      break;

    QString path = imageList[i];
    progress.setValue(i);

    // 根据选择执行相应的处理
    if (selected == "星点检测" || selected == "全部处理") {
      cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
      if (!img.empty()) {
        detectedStars = starDetector.multiscale_detect_stars(img);
        exportDetectedStarsToCSV(path + "_stars.csv");
      }
    }
    if (selected == "自动拉伸" || selected == "全部处理") {
      // 处理并保存
      cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
      if (!img.empty()) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        cv::Mat processed;
        img.convertTo(processed, -1, 255.0 / (maxVal - minVal),
                      -minVal * 255.0 / (maxVal - minVal));
        cv::imwrite((path.toStdString() + "_stretched.png"), processed);
      }
    }
    // ... 其他处理选项
  }

  progress.setValue(imageList.size());
}

void ImagePreviewDialog::saveAsFITS() {
  if (originalImage.empty())
    return;

  QString fileName = QFileDialog::getSaveFileName(
      this, "保存为FITS", imageList[currentIndex], "FITS文件 (*.fits)");

  if (!fileName.isEmpty()) {
    saveMatToFits(originalImage, fileName.toStdString());
  }
}

void ImagePreviewDialog::toggleAutoDetection() {
  autoDetectionEnabled = !autoDetectionEnabled;
  if (autoDetectionEnabled && !originalImage.empty()) {
    startStarDetection();
  }
}

void ImagePreviewDialog::exportStarData() {
  if (detectedStars.empty()) {
    QMessageBox::warning(this, "警告", "没有检测到星点数据");
    return;
  }

  QString fileName = QFileDialog::getSaveFileName(
      this, "导出星点数据", imageList[currentIndex] + "_stars.csv",
      "CSV文件 (*.csv)");

  if (!fileName.isEmpty()) {
    exportDetectedStarsToCSV(fileName);
  }
}

void ImagePreviewDialog::exportDetectedStarsToCSV(const QString &path) {
  QFile file(path);
  if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
    QMessageBox::warning(this, "错误", "无法创建文件");
    return;
  }

  QTextStream out(&file);
  out << "X,Y,亮度\n";

  for (const auto &star : detectedStars) {
    // 获取星点位置的亮度值
    cv::Point pt(star.x, star.y);
    uchar brightness = originalImage.at<uchar>(pt);
    out << star.x << "," << star.y << "," << (int)brightness << "\n";
  }
}

void ImagePreviewDialog::showHistogram() {
  if (originalImage.empty()) {
    QMessageBox::warning(this, "警告", "没有可用的图像");
    return;
  }

  try {
    histogramDialog->showHistogram(originalImage);
    histogramDialog->show();
  } catch (const std::exception &e) {
    QMessageBox::critical(this, "错误",
                          QString("无法显示直方图: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::updateHistogram() {
  if (originalImage.empty())
    return;

  if (histogramDialog->isVisible()) {
    try {
      histogramDialog->showHistogram(originalImage);
    } catch (const std::exception &e) {
      spdlog::error("更新直方图失败: {}", e.what());
    }
  }
}

void ImagePreviewDialog::processImage(const QString &path) {
  cv::Mat image = cv::imread(path.toStdString(), cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    spdlog::error("无法加载图像: {}", path.toStdString());
    return;
  }

  // 根据图像类型选择合适的保存方式
  if (path.endsWith(".fits", Qt::CaseInsensitive)) {
    saveMatToFits(image, path.toStdString());
  } else if (image.depth() == CV_16U) {
    saveMatTo16BitPng(image, path.toStdString());
  } else {
    saveMatTo8BitJpg(image, path.toStdString());
  }
}

double ImagePreviewDialog::calculateImageStatistics() {
  if (originalImage.empty())
    return 0.0;

  cv::Scalar mean, stddev;
  cv::meanStdDev(originalImage, mean, stddev);

  double snr = mean[0] / stddev[0];
  spdlog::info("图像统计: 均值={:.2f}, 标准差={:.2f}, SNR={:.2f}", mean[0],
               stddev[0], snr);

  return snr;
}

void ImagePreviewDialog::showExifInfo() {
  if (exifData.empty()) {
    QMessageBox::information(this, "EXIF信息", "该图像没有EXIF信息");
    return;
  }

  QString info;
  for (const auto &exif : exifData) {
    info += QString("%1: %2\n")
                .arg(QString::fromStdString(exif.tag_name),
                     QString::fromStdString(std::visit(
                         [](auto &&arg) {
                           using T = std::decay_t<decltype(arg)>;
                           if constexpr (std::is_same_v<T, std::string>) {
                             return arg;
                           } else if constexpr (std::is_arithmetic_v<T>) {
                             return std::to_string(arg);
                           } else {
                             return std::string("复合数据");
                           }
                         },
                         exif.value)));
  }

  QDialog dialog(this);
  dialog.setWindowTitle("EXIF信息");
  auto layout = new QVBoxLayout(&dialog);
  auto text = new QTextEdit;
  text->setPlainText(info);
  text->setReadOnly(true);
  layout->addWidget(text);
  dialog.exec();
}

void ImagePreviewDialog::applyConvolution() {
  // 创建卷积配置对话框
  QDialog dialog(this);
  dialog.setWindowTitle("卷积设置");
  auto layout = new QVBoxLayout(&dialog);

  // 添加卷积参数控件
  auto kernelSizeSpinBox = new QSpinBox(&dialog);
  kernelSizeSpinBox->setRange(3, 15);
  kernelSizeSpinBox->setSingleStep(2);
  kernelSizeSpinBox->setValue(3);
  layout->addWidget(new QLabel("核大小:"));
  layout->addWidget(kernelSizeSpinBox);

  // 添加确定和取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal, &dialog);
  layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  if (dialog.exec() == QDialog::Accepted) {
    ConvolutionConfig config;
    config.kernel_size = kernelSizeSpinBox->value();
    config.kernel =
        std::vector<float>(config.kernel_size * config.kernel_size, 1.0f);
    processConvolution(config);
  }
}

void ImagePreviewDialog::processConvolution(const ConvolutionConfig &config) {
  if (originalImage.empty())
    return;

  try {
    cv::Mat result = Convolve::process(originalImage, config);
    QImage qImage(result.data, result.cols, result.rows, result.step,
                  QImage::Format_Grayscale8);
    imageLabel->setPixmap(QPixmap::fromImage(qImage));
    updateImage();
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "错误",
                         QString("卷积处理失败: %1").arg(e.what()));
  }
}

void ImagePreviewDialog::applyDeconvolution() {
  QDialog dialog(this);
  dialog.setWindowTitle("反卷积设置");
  auto layout = new QVBoxLayout(&dialog);

  auto methodCombo = new QComboBox(&dialog);
  methodCombo->addItems({"Richardson-Lucy", "Wiener", "Tikhonov"});
  layout->addWidget(new QLabel("方法:"));
  layout->addWidget(methodCombo);

  auto iterSpinBox = new QSpinBox(&dialog);
  iterSpinBox->setRange(1, 100);
  iterSpinBox->setValue(30);
  layout->addWidget(new QLabel("迭代次数:"));
  layout->addWidget(iterSpinBox);

  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal, &dialog);
  layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  if (dialog.exec() == QDialog::Accepted) {
    DeconvolutionConfig config;
    config.method = static_cast<DeconvMethod>(methodCombo->currentIndex());
    config.iterations = iterSpinBox->value();
    processDeconvolution(config);
  }
}

void ImagePreviewDialog::processDeconvolution(
    const DeconvolutionConfig &config) {
  if (originalImage.empty())
    return;

  try {
    cv::Mat result = Convolve::process(originalImage, config);
    QImage qImage(result.data, result.cols, result.rows, result.step,
                  QImage::Format_Grayscale8);
    imageLabel->setPixmap(QPixmap::fromImage(qImage));
    updateImage();
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "错误",
                         QString("反卷积处理失败: %1").arg(e.what()));
  }
}

// 新增：生成缩略图网格并在对话框中显示
void ImagePreviewDialog::createThumbnailGrid() {
  QDialog gridDialog(this);
  gridDialog.setWindowTitle("缩略图网格预览");
  QGridLayout *gridLayout = new QGridLayout(&gridDialog);

  int cols = 4;
  int row = 0, col = 0;
  for (const QString &imgPath : imageList) {
    QLabel *thumbLabel = new QLabel;
    QPixmap pix;
    // 使用 FITS 文件时调用 IO 模块加载图像
    if (imgPath.endsWith(".fits", Qt::CaseInsensitive)) {
      cv::Mat mat = ::loadImage(imgPath.toStdString());
      if (!mat.empty()) {
        QImage qImg(mat.data, mat.cols, mat.rows, mat.step,
                    QImage::Format_Grayscale8);
        pix = QPixmap::fromImage(qImg).scaled(100, 100, Qt::KeepAspectRatio,
                                              Qt::SmoothTransformation);
      }
    } else {
      QImage image(imgPath);
      if (!image.isNull())
        pix = QPixmap::fromImage(image).scaled(100, 100, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation);
    }
    thumbLabel->setPixmap(pix);
    gridLayout->addWidget(thumbLabel, row, col);
    if (++col >= cols) {
      col = 0;
      row++;
    }
  }

  gridDialog.exec();
}

void ImagePreviewDialog::applyBatchOperations() {
  // 创建批处理设置对话框
  QDialog dialog(this);
  dialog.setWindowTitle("批处理设置");
  auto layout = new QVBoxLayout(&dialog);

  // 创建批处理选项
  QCheckBox *autoStretchCheck = new QCheckBox("自动拉伸", &dialog);
  QCheckBox *denoisingCheck = new QCheckBox("降噪", &dialog);
  QCheckBox *edgeDetectionCheck = new QCheckBox("边缘检测", &dialog);
  QCheckBox *starDetectionCheck = new QCheckBox("星点检测", &dialog);

  layout->addWidget(autoStretchCheck);
  layout->addWidget(denoisingCheck);
  layout->addWidget(edgeDetectionCheck);
  layout->addWidget(starDetectionCheck);

  // 添加确定和取消按钮
  auto buttonBox = new QDialogButtonBox(
      QDialogButtonBox::Ok | QDialogButtonBox::Cancel, Qt::Horizontal, &dialog);
  layout->addWidget(buttonBox);

  connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

  if (dialog.exec() == QDialog::Accepted) {
    // 更新批处理配置
    batchConfig.autoStretch = autoStretchCheck->isChecked();
    batchConfig.denoising = denoisingCheck->isChecked();
    batchConfig.edgeDetection = edgeDetectionCheck->isChecked();

    // 显示进度对话框
    QProgressDialog progress("正在批处理...", "取消", 0, imageList.size(),
                             this);
    progress.setWindowModality(Qt::WindowModal);

    // 对每个图像执行批处理
    for (int i = 0; i < imageList.size(); ++i) {
      if (progress.wasCanceled())
        break;

      progress.setValue(i);
      QString currentImage = imageList[i];

      try {
        // 加载图像
        cv::Mat img =
            cv::imread(currentImage.toStdString(), cv::IMREAD_UNCHANGED);
        if (img.empty())
          continue;

        // 应用选中的处理操作
        if (batchConfig.autoStretch) {
          double minVal, maxVal;
          cv::minMaxLoc(img, &minVal, &maxVal);
          img.convertTo(img, -1, 255.0 / (maxVal - minVal),
                        -minVal * 255.0 / (maxVal - minVal));
        }

        if (batchConfig.denoising) {
          cv::Mat denoised;
          cv::fastNlMeansDenoising(img, denoised);
          img = denoised;
        }

        if (batchConfig.edgeDetection) {
          cv::Mat edges;
          cv::Canny(img, edges, 100, 200);
          img = edges;
        }

        // 保存处理后的图像
        QString outputPath = currentImage;
        outputPath.insert(outputPath.lastIndexOf('.'), "_processed");
        cv::imwrite(outputPath.toStdString(), img);

        // 如果选择了星点检测
        if (starDetectionCheck->isChecked()) {
          StarDetectionConfig config;
          auto stars = starDetector.multiscale_detect_stars(img);
          QString starDataPath = outputPath;
          starDataPath.replace(starDataPath.lastIndexOf('.'), 4, "_stars.csv");
          exportDetectedStarsToCSV(starDataPath);
        }

      } catch (const std::exception &e) {
        spdlog::error("处理图像失败 {}: {}", currentImage.toStdString(),
                      e.what());
        continue;
      }
    }

    progress.setValue(imageList.size());
    QMessageBox::information(this, "完成", "批处理操作已完成");
  }
}

void ImagePreviewDialog::showStatistics() {
  if (originalImage.empty()) {
    QMessageBox::warning(this, "错误", "没有可用的图像");
    return;
  }

  try {
    // 计算基本统计信息
    cv::Scalar mean, stddev;
    cv::meanStdDev(originalImage, mean, stddev);

    // 计算最大最小值
    double minVal, maxVal;
    cv::minMaxLoc(originalImage, &minVal, &maxVal);

    // 计算直方图
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&originalImage, 1, 0, cv::Mat(), hist, 1, &histSize,
                 &histRange);

    // 找到直方图峰值
    double maxHistVal;
    cv::minMaxLoc(hist, nullptr, &maxHistVal);

    // 计算信噪比
    double snr = mean[0] / stddev[0];

    // 创建统计信息对话框
    QDialog statsDialog(this);
    statsDialog.setWindowTitle("图像统计信息");
    auto layout = new QVBoxLayout(&statsDialog);

    // 显示统计数据
    QString statsText =
        QString("图像尺寸: %1 x %2\n"
                "位深度: %3\n"
                "均值: %.2f\n"
                "标准差: %.2f\n"
                "最小值: %.2f\n"
                "最大值: %.2f\n"
                "信噪比(SNR): %.2f\n"
                "直方图峰值: %.0f")
            .arg(originalImage.cols)
            .arg(originalImage.rows)
            .arg(originalImage.depth() * originalImage.channels())
            .arg(mean[0])
            .arg(stddev[0])
            .arg(minVal)
            .arg(maxVal)
            .arg(snr)
            .arg(maxHistVal);

    auto statsLabel = new QLabel(statsText);
    layout->addWidget(statsLabel);

    // 添加关闭按钮
    auto closeButton = new QPushButton("关闭", &statsDialog);
    connect(closeButton, &QPushButton::clicked, &statsDialog, &QDialog::accept);
    layout->addWidget(closeButton);

    statsDialog.exec();

  } catch (const std::exception &e) {
    QMessageBox::critical(this, "错误",
                          QString("计算统计信息失败: %1").arg(e.what()));
  }
}
