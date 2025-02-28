set(VERSION_SOURCES
    VersionControl.cpp
    CacheManager.cpp
    BranchManager.cpp
    TagManager.cpp
    MetadataManager.cpp
    ImageOperations.cpp
)

set(VERSION_HEADERS
    VersionControl.hpp
    VersionTypes.hpp
    CacheManager.hpp
)

# 创建版本控制库
add_library(version STATIC ${VERSION_SOURCES} ${VERSION_HEADERS})

target_include_directories(version
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接所需的依赖库
target_link_libraries(version
    PUBLIC
    ${OpenCV_LIBS}
    ZLIB::ZLIB
    OpenSSL::SSL
    OpenSSL::Crypto
    spdlog::spdlog
)

# 确保使用C++17标准
target_compile_features(version PUBLIC cxx_std_17)

# 如果使用MSVC编译器
if(MSVC)
    target_compile_options(version PRIVATE /W4 /WX)
else()
    # GCC或Clang编译器
    target_compile_options(version PRIVATE -Wall -Wextra -Werror)
endif()

# 源码分组
source_group(
    TREE ${CMAKE_CURRENT_SOURCE_DIR}
    FILES ${VERSION_SOURCES} ${VERSION_HEADERS}
)
