#include <QtWebEngineWidgets>
#include <QtWidgets>
#include <QtPositioning>
#include <numbers>
#include <optional>
#include <spdlog/spdlog.h>
#include <variant>

using Coordinate = std::pair<double, double>;
using Address = QString;

enum class LocationStatus {
    Inactive,
    Acquiring,
    Available,
    Error
};

class GeoMapWidget : public QWebEngineView {
    Q_OBJECT
public:
    explicit GeoMapWidget(QWidget *parent = nullptr)
        : QWebEngineView(parent), m_zoomLevel(12) {
        setupWebChannel();
        setupPositionSource();
        loadMap();
    }

    void setCenter(const Coordinate &coord) noexcept {
        try {
            runJavaScript(
                QString("setCenter(%1, %2);").arg(coord.first).arg(coord.second));
        } catch (const std::exception &e) {
            spdlog::error("Failed to set center: {}", e.what());
        }
    }

    [[nodiscard]] Coordinate currentPosition() const { return m_currentPos; }

    void startPositioning() {
        if (m_posSource) {
            m_posSource->startUpdates();
            emit locationStatusChanged(LocationStatus::Acquiring);
        }
    }

    void stopPositioning() {
        if (m_posSource) {
            m_posSource->stopUpdates();
            emit locationStatusChanged(LocationStatus::Inactive);
        }
    }

signals:
    void positionChanged(const Coordinate &coord);
    void addressResolved(const Address &addr);
    void locationStatusChanged(LocationStatus status);
    void accuracyChanged(double accuracy);

private:
    void setupWebChannel() {
        m_webChannel = new QWebChannel(this);
        m_webChannel->registerObject(QStringLiteral("cppBridge"), this);
        page()->setWebChannel(m_webChannel);
    }

    void setupPositionSource() {
        m_posSource = QGeoPositionInfoSource::createDefaultSource(this);
        if (m_posSource) {
            connect(m_posSource, &QGeoPositionInfoSource::positionUpdated,
                    this, &GeoMapWidget::handlePositionUpdate);
            connect(m_posSource, &QGeoPositionInfoSource::errorOccurred,
                    this, &GeoMapWidget::handlePositionError);
            
            m_posSource->setPreferredPositioningMethods(
                QGeoPositionInfoSource::AllPositioningMethods);
        } else {
            spdlog::error("No positioning source available");
            emit locationStatusChanged(LocationStatus::Error);
        }
    }

    void handlePositionUpdate(const QGeoPositionInfo &info) {
        auto pos = info.coordinate();
        m_currentPos = {pos.latitude(), pos.longitude()};
        setCenter(m_currentPos);
        emit positionChanged(m_currentPos);
        emit locationStatusChanged(LocationStatus::Available);
        
        if (info.hasAttribute(QGeoPositionInfo::HorizontalAccuracy)) {
            emit accuracyChanged(info.attribute(QGeoPositionInfo::HorizontalAccuracy));
        }
    }

    void handlePositionError(QGeoPositionInfoSource::Error error) {
        spdlog::error("Position source error: {}", error);
        emit locationStatusChanged(LocationStatus::Error);
    }

    void loadMap() {
        const QString html = R"(
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
                <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            </head>
            <body>
                <div id="map" style="height: 100%;"></div>
                <script>
                    var map = L.map('map').setView([51.505, -0.09], %1);
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '© OpenStreetMap'
                    }).addTo(map);

                    var marker = L.marker([51.505, -0.09]).addTo(map);
                    var accuracyCircle;

                    function updatePosition(lat, lng, accuracy) {
                        marker.setLatLng([lat, lng]);
                        map.panTo([lat, lng]);
                        
                        if (accuracy && accuracy > 0) {
                            if (accuracyCircle) {
                                accuracyCircle.setLatLng([lat, lng]).setRadius(accuracy);
                            } else {
                                accuracyCircle = L.circle([lat, lng], {
                                    radius: accuracy,
                                    color: 'blue',
                                    fillColor: '#3388ff',
                                    fillOpacity: 0.1
                                }).addTo(map);
                            }
                        }
                    }

                    var cppBridge;
                    new QWebChannel(qt.webChannelTransport, function(channel) {
                        cppBridge = channel.objects.cppBridge;
                    });

                    map.on('click', function(e) {
                        cppBridge.onMapClicked(e.latlng.lat, e.latlng.lng);
                    });
                </script>
            </body>
            </html>
        )".arg(m_zoomLevel);

        setHtml(html);
    }

    Q_INVOKABLE void onMapClicked(double lat, double lng) {
        m_currentPos = {lat, lng};
        emit positionChanged(m_currentPos);
        reverseGeocode(m_currentPos);
    }

    void reverseGeocode(const Coordinate &coord) {
        QNetworkRequest request(QUrl(QString("https://nominatim.openstreetmap.org/"
                                           "reverse?format=json&lat=%1&lon=%2")
                                       .arg(coord.first)
                                       .arg(coord.second)));
        request.setRawHeader("User-Agent", "GeoLocationApp/1.0");

        QNetworkReply *reply = m_networkMgr.get(request);
        connect(reply, &QNetworkReply::finished, [this, reply]() {
            if (reply->error() == QNetworkReply::NoError) {
                const auto doc = QJsonDocument::fromJson(reply->readAll());
                const auto addr = doc.object()["display_name"].toString();
                emit addressResolved(addr);
            } else {
                spdlog::warn("Reverse geocoding failed: {}",
                           reply->errorString().toStdString());
            }
            reply->deleteLater();
        });
    }

    QWebChannel *m_webChannel{nullptr};
    QNetworkAccessManager m_networkMgr;
    QGeoPositionInfoSource *m_posSource{nullptr};
    Coordinate m_currentPos;
    int m_zoomLevel;
};

class GeoMath {
public:
  // 使用Haversine公式计算距离（米）
  [[nodiscard]] static double calculateDistance(const Coordinate &p1,
                                                const Coordinate &p2) noexcept {
    using namespace std::numbers;
    const auto [lat1, lon1] = p1;
    const auto [lat2, lon2] = p2;

    const double dLat = (lat2 - lat1) * pi / 180.0;
    const double dLon = (lon2 - lon1) * pi / 180.0;

    const double a = pow(sin(dLat / 2), 2) + cos(lat1 * pi / 180.0) *
                                                 cos(lat2 * pi / 180.0) *
                                                 pow(sin(dLon / 2), 2);

    return 2 * atan2(sqrt(a), sqrt(1 - a)) * earthRadius;
  }

private:
  static constexpr double earthRadius = 6371e3; // 米
};

class LocationSettingsPanel : public QWidget {
    Q_OBJECT
public:
    explicit LocationSettingsPanel(QWidget *parent = nullptr) : QWidget(parent) {
        setupUI();
        connectSignals();
    }

private:
    void setupUI() {
        QVBoxLayout *mainLayout = new QVBoxLayout(this);
        
        // 地图控件
        m_mapWidget = new GeoMapWidget;
        mainLayout->addWidget(m_mapWidget);

        // 坐标输入区域
        QFormLayout *coordLayout = new QFormLayout;
        m_latEdit = new QLineEdit;
        m_lonEdit = new QLineEdit;
        coordLayout->addRow("Latitude:", m_latEdit);
        coordLayout->addRow("Longitude:", m_lonEdit);
        mainLayout->addLayout(coordLayout);

        // 定位控制区域
        QHBoxLayout *locationCtrlLayout = new QHBoxLayout;
        
        m_locateBtn = new QPushButton("定位到我的位置");
        connect(m_locateBtn, &QPushButton::clicked, this, &LocationSettingsPanel::togglePositioning);
        locationCtrlLayout->addWidget(m_locateBtn);

        m_statusLabel = new QLabel("未定位");
        locationCtrlLayout->addWidget(m_statusLabel);
        
        m_accuracyLabel = new QLabel;
        locationCtrlLayout->addWidget(m_accuracyLabel);
        
        mainLayout->addLayout(locationCtrlLayout);

        // 保存按钮
        QPushButton *saveBtn = new QPushButton("保存位置");
        connect(saveBtn, &QPushButton::clicked, this, &LocationSettingsPanel::saveSettings);
        mainLayout->addWidget(saveBtn);
    }

    void connectSignals() {
        connect(m_mapWidget, &GeoMapWidget::positionChanged,
                [this](const Coordinate &coord) {
                    m_latEdit->setText(QString::number(coord.first, 'f', 6));
                    m_lonEdit->setText(QString::number(coord.second, 'f', 6));
                });

        connect(m_mapWidget, &GeoMapWidget::locationStatusChanged,
                [this](LocationStatus status) {
                    switch (status) {
                        case LocationStatus::Inactive:
                            m_statusLabel->setText("未定位");
                            m_locateBtn->setText("开始定位");
                            break;
                        case LocationStatus::Acquiring:
                            m_statusLabel->setText("定位中...");
                            m_locateBtn->setText("停止定位");
                            break;
                        case LocationStatus::Available:
                            m_statusLabel->setText("已定位");
                            m_locateBtn->setText("停止定位");
                            break;
                        case LocationStatus::Error:
                            m_statusLabel->setText("定位错误");
                            m_locateBtn->setText("重试定位");
                            break;
                    }
                });

        connect(m_mapWidget, &GeoMapWidget::accuracyChanged,
                [this](double accuracy) {
                    m_accuracyLabel->setText(QString("精度: %1米").arg(accuracy, 0, 'f', 1));
                });
    }

    void togglePositioning() {
        if (m_locateBtn->text() == "开始定位" || m_locateBtn->text() == "重试定位") {
            m_mapWidget->startPositioning();
        } else {
            m_mapWidget->stopPositioning();
        }
    }

    void saveSettings() {
        try {
            const double lat = m_latEdit->text().toDouble();
            const double lon = m_lonEdit->text().toDouble();

            if (!isValidCoordinate(lat, lon)) {
                throw std::runtime_error("无效的坐标值");
            }

            QSettings settings;
            settings.beginGroup("Location");
            settings.setValue("latitude", lat);
            settings.setValue("longitude", lon);
            settings.endGroup();

            spdlog::info("Location saved: ({}, {})", lat, lon);
            QMessageBox::information(this, "成功", "位置保存成功！");

        } catch (const std::exception &e) {
            spdlog::error("Save failed: {}", e.what());
            QMessageBox::warning(this, "错误", e.what());
        }
    }

    [[nodiscard]] static bool isValidCoordinate(double lat, double lon) noexcept {
        return (lat >= -90.0 && lat <= 90.0) && (lon >= -180.0 && lon <= 180.0);
    }

    GeoMapWidget *m_mapWidget;
    QLineEdit *m_latEdit;
    QLineEdit *m_lonEdit;
    QPushButton *m_locateBtn;
    QLabel *m_statusLabel;
    QLabel *m_accuracyLabel;
};

#include "GeoLocation.moc"