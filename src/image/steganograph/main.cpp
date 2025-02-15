#include "StegoWindow.hpp"
#include <QApplication>

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  
  // 设置应用程序信息
  QApplication::setApplicationName("图像隐写工具");
  QApplication::setApplicationVersion("1.0");
  
  // 创建并显示主窗口
  StegoWindow window;
  window.show();
  
  return app.exec();
}
