#ifndef UPDATECHECK_H
#define UPDATECHECK_H

#include <QtNetwork>
#include <QtWidgets>
#include <semver.hpp>


// Result 类代替 std::expected
template <typename T> class Result {
public:
  Result(const T &value) : m_value(value), m_isError(false) {}
  Result(const QString &error) : m_error(error), m_isError(true) {}

  bool hasError() const { return m_isError; }
  const QString &error() const { return m_error; }
  const T &value() const { return m_value; }

private:
  T m_value;
  QString m_error;
  bool m_isError;
};

class UpdateChecker : public QObject {
  Q_OBJECT
public:
  explicit UpdateChecker(QObject *parent = nullptr);
  void checkForUpdates(bool force = false);

signals:
  void updateAvailable(const QJsonObject &updateInfo);
  void updateNotAvailable();
  void updateError(const QString &message);

private slots:
  void onUpdateInfoReceived(QNetworkReply *reply);

private:
  QNetworkAccessManager *manager;
  QSettings settings;

  bool shouldCheck() const;
  Result<QJsonObject> parseUpdateInfo(const QByteArray &data) noexcept;
  void handleNetworkError(QNetworkReply::NetworkError code);
};

class UpdateDialog : public QDialog {
  Q_OBJECT
public:
  explicit UpdateDialog(const QJsonObject &updateInfo,
                        QWidget *parent = nullptr);

private:
  void setupUI(const QJsonObject &info);
  QString formatChangelog(const QString &raw) const;
  void loadUserPreferences();
  void saveUserPreferences();
  void closeEvent(QCloseEvent *event) override;

private slots:
  void onIgnoreClicked();
  void onLaterClicked();
  void onUpdateClicked();
};

#endif // UPDATECHECK_H