# Reference: https://doc.qt.io/qt-6/cmake-manual.html

cmake_minimum_required(VERSION 3.16)

# Add folder where are supportive functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Include Qt basic functions
include(QtCommon)

# Basic information about project

project(example VERSION 1.0)

# Set PROJECT_VERSION_PATCH and PROJECT_VERSION_TWEAK to 0 if not present, needed by add_project_meta
fix_project_version()

# Set additional project information
set(COMPANY "Example")
set(COPYRIGHT "Copyright (c) 2014 Vincent Lee. All rights reserved.")
set(IDENTIFIER "com.example.Example")

set(SOURCE_FILES
    src/main.cpp
    src/mainwindow.cpp
    src/CrashDialog.cpp
    src/CrashHandler.cpp
    src/HistogramDialog.cpp
    src/ImageContextMenu.cpp
    src/ImagePreviewDialog.cpp
    src/MetaDataDialog.cpp
    src/UpdateCheck.cpp

    src/image/Convolve.cpp
    src/image/Crop.cpp
    src/image/Denoise.cpp
    src/image/Exif.cpp
    src/image/Filter.cpp
    src/image/FWHM.cpp
    src/image/HFR.cpp
    src/image/Histogram.cpp
    src/image/ImageIO.cpp
    src/image/MetaData.cpp
    src/image/SIMDHelper.cpp
    src/image/Stack.cpp
    src/image/StarDetector.cpp
    src/image/VersionControl.cpp

    src/json/JsonEditor.cpp
    src/json/JsonModel.cpp
    src/json/JsonSyntaxHighlighter.cpp

    src/log/LogAnalyzer.cpp
    src/log/LogChartModel.cpp
    src/log/LogFilterModel.cpp
    src/log/LogPanel.cpp
    src/log/LogSearchCompleter.cpp

    src/utils/ThreadPool.cpp
)

add_project_meta(META_FILES_TO_INCLUDE)

set(RESOURCE_FILES example.qrc)

find_package(Qt6 COMPONENTS Widgets Network Charts Concurrent Positioning WebView REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(TBB REQUIRED)
find_package(CFITSIO REQUIRED)

qt_standard_project_setup()

add_executable(${PROJECT_NAME} WIN32 MACOSX_BUNDLE
    ${SOURCE_FILES} ${META_FILES_TO_INCLUDE} ${RESOURCE_FILES}
    src/image/ImageManager.cpp
    src/image/ImageManager.h
)

target_link_directories(${PROJECT_NAME} PRIVATE ${OpenCV_LIB_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
)

target_precompile_headers(${PROJECT_NAME} INTERFACE QtWidgets.h)

target_link_libraries(${PROJECT_NAME} 
    Qt6::Widgets fmt::fmt Qt6::Network Qt6::Charts Qt6::Concurrent Qt6::Positioning
    Qt6::WebView
    ${OpenCV_LIBS} ZLIB::ZLIB OpenSSL::SSL OpenSSL::Crypto TBB::tbb
    ${CFITSIO_LIBRARIES})

if (WIN32)
target_link_libraries(${PROJECT_NAME} dbghelp)
endif()

install(
    TARGETS ${PROJECT_NAME}
    BUNDLE DESTINATION /Applications
)
