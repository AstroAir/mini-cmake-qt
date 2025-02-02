#pragma once
#include <string>
#include <functional>

#if defined(USE_BOOST_STACKTRACE)
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace CrashHandler {
struct Config {
    bool enableMinidump = true;
    bool enableStackTrace = true;
    std::string crashLogPath;
    std::function<void(const std::string&)> customCallback;
};

void init(const Config& config = Config());
void setCustomHandler(std::function<void(const std::string&)> handler);
std::string getStackTrace(size_t maxFrames = 62);
std::string getPlatformInfo();
std::string getMemoryInfo();
std::string getCpuInfo();
void writeMiniDump(const std::string& path);
} // namespace CrashHandler