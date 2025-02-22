#pragma once

#include <QWidget>
#include <QMap>

class QVBoxLayout;
class QComboBox;
class QStackedWidget;
class QPushButton;

class EffectPanel : public QWidget {
  Q_OBJECT

public:
  explicit EffectPanel(QWidget *parent = nullptr);

signals:
  void effectApplied();

public slots:
  void updateEffectParams();
  void applyCurrentEffect();
  void previewEffect();

private:
  void setupUI();
  void createEffectControls();
  
  QVBoxLayout *m_layout;
  QComboBox *m_effectSelect;
  QStackedWidget *m_paramStack;
  QPushButton *m_applyButton;
  QPushButton *m_previewButton;
  
  QMap<QString, QWidget*> m_effectParams;
};
