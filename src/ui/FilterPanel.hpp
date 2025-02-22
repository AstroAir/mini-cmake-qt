#pragma once

#include <QWidget>
#include <QMap>

class QVBoxLayout;
class QComboBox;
class QSlider;
class QPushButton;
class QLabel;

class FilterPanel : public QWidget {
  Q_OBJECT

public:
  explicit FilterPanel(QWidget *parent = nullptr);

signals:
  void filterApplied();

public slots:
  void updateFilterParams();
  void applyCurrentFilter();
  void resetFilters();

private:
  void setupUI();
  void createFilterControls();
  void updatePreview();

  QVBoxLayout *m_layout;
  QComboBox *m_filterSelect;
  QMap<QString, QSlider*> m_paramSliders;
  QPushButton *m_applyButton;
  QPushButton *m_resetButton;
  QLabel *m_previewLabel;
};
