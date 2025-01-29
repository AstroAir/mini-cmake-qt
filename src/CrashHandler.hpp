#pragma once
#include <string>

namespace CrashHandler {
void init();
std::string getStackTrace();
std::string getPlatformInfo();
} // namespace CrashHandler