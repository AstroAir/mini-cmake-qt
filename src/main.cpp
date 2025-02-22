#include "mainwindow.h"
#include "utils/CrashHandler.hpp"
#include <QApplication>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h> // 添加文件日志记录器头文件

int main(int argc, char *argv[]) {
  // 初始化崩溃处理器
  CrashHandler::init();
  
  // 创建文件日志记录器
  auto file_logger = spdlog::basic_logger_mt("file_logger", "logs/app.log");
  spdlog::set_default_logger(file_logger); // Set file logger as the default logger 
  spdlog::set_level(spdlog::level::debug); // Set global log level to debug
  spdlog::info("Application started");

  QApplication app(argc, argv);
  MainWindow mainWindow;
  mainWindow.show();

  spdlog::info("Main window shown");
  return app.exec();
}
