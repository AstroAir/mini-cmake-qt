#include "CrashHandler.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

#if defined(_WIN32)
// clang-format off
#include <windows.h>
#include <dbghelp.h>
#include <psapi.h>
// clang-format on
#pragma comment(lib, "dbghelp.lib")
#elif defined(__APPLE__)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <mach-o/dyld.h>
#else
#include <cxxabi.h>
#include <execinfo.h>
#include <link.h>

#endif

namespace {

// 公共工具函数
std::string timestamp() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
  return oss.str();
}

// 平台特定实现
#if defined(_WIN32)

const int max_frames = 62;
LONG WINAPI exceptionHandler(PEXCEPTION_POINTERS pExceptionInfo) {
  std::stringstream ss;

  // 收集基本信息
  ss << "Crash Time: " << timestamp() << "\n";
  ss << "Exception Code: 0x" << std::hex
     << pExceptionInfo->ExceptionRecord->ExceptionCode << "\n";

  // 获取堆栈跟踪
  HANDLE process = GetCurrentProcess();
  SymInitialize(process, NULL, TRUE);

  void *stack[max_frames];
  WORD frames = CaptureStackBackTrace(0, max_frames, stack, NULL);

  SYMBOL_INFO *symbol =
      (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
  symbol->MaxNameLen = 255;
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

  for (WORD i = 0; i < frames; ++i) {
    SymFromAddr(process, (DWORD64)stack[i], 0, symbol);
    ss << i << " " << symbol->Name << " (0x" << std::hex << symbol->Address
       << ")\n";
  }

  free(symbol);
  SymCleanup(process);

  // 写入文件
  FILE *f = fopen("crash.log", "w");
  if (f) {
    fwrite(ss.str().c_str(), 1, ss.str().size(), f);
    fclose(f);
  }

  return EXCEPTION_EXECUTE_HANDLER;
}

#elif defined(__APPLE__)

void signalHandler(int sig) {
  std::stringstream ss;
  ss << "Signal: " << sig << " (" << strsignal(sig) << ")\n";
  ss << "Time: " << timestamp() << "\n\n";

  void *callstack[128];
  int frames = backtrace(callstack, 128);
  char **symbols = backtrace_symbols(callstack, frames);

  for (int i = 0; i < frames; ++i) {
    Dl_info info;
    if (dladdr(callstack[i], &info)) {
      int status;
      char *demangled =
          abi::__cxa_demangle(info.dli_sname, NULL, NULL, &status);
      ss << "#" << i << " 0x" << std::hex << callstack[i] << " "
         << (demangled ? demangled : info.dli_sname) << " + "
         << (char *)callstack[i] - (char *)info.dli_saddr << "\n";
      free(demangled);
    } else {
      ss << "#" << i << " " << symbols[i] << "\n";
    }
  }
  free(symbols);

  FILE *f = fopen("crash.log", "w");
  if (f) {
    fwrite(ss.str().c_str(), 1, ss.str().size(), f);
    fclose(f);
  }

  exit(1);
}

#else // Linux

void signalHandler(int sig) {
  std::stringstream ss;
  ss << "Signal: " << sig << " (" << strsignal(sig) << ")\n";
  ss << "Time: " << timestamp() << "\n\n";

  void *array[50];
  int size = backtrace(array, 50);
  char **symbols = backtrace_symbols(array, size);

  for (int i = 0; i < size; ++i) {
    Dl_info info;
    if (dladdr(array[i], &info)) {
      int status;
      char *demangled =
          abi::__cxa_demangle(info.dli_sname, NULL, NULL, &status);
      ss << "#" << i << " 0x" << std::hex << array[i] << " "
         << (demangled ? demangled : info.dli_sname) << " + "
         << (char *)array[i] - (char *)info.dli_saddr << "\n";
      free(demangled);
    } else {
      ss << "#" << i << " " << symbols[i] << "\n";
    }
  }
  free(symbols);

  FILE *f = fopen("crash.log", "w");
  if (f) {
    fwrite(ss.str().c_str(), 1, ss.str().size(), f);
    fclose(f);
  }

  exit(1);
}

#endif

} // namespace

namespace CrashHandler {

void init() {
#if defined(_WIN32)
  SetUnhandledExceptionFilter(exceptionHandler);
#else
  signal(SIGSEGV, signalHandler);
  signal(SIGABRT, signalHandler);
  signal(SIGFPE, signalHandler);
  signal(SIGILL, signalHandler);
  signal(SIGBUS, signalHandler);
#endif
}

std::string getPlatformInfo() {
  std::stringstream ss;

#if defined(_WIN32)
  OSVERSIONINFOEX osvi;
  ZeroMemory(&osvi, sizeof(OSVERSIONINFOEX));
  osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
  GetVersionEx((OSVERSIONINFO *)&osvi);

  ss << "Windows Version: " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion
     << "." << osvi.dwBuildNumber << "\n";

  SYSTEM_INFO sysInfo;
  GetNativeSystemInfo(&sysInfo);
  ss << "Processor Architecture: ";
  switch (sysInfo.wProcessorArchitecture) {
  case PROCESSOR_ARCHITECTURE_AMD64:
    ss << "x64";
    break;
  case PROCESSOR_ARCHITECTURE_ARM:
    ss << "ARM";
    break;
  case PROCESSOR_ARCHITECTURE_IA64:
    ss << "IA64";
    break;
  case PROCESSOR_ARCHITECTURE_INTEL:
    ss << "x86";
    break;
  default:
    ss << "Unknown";
  }

#elif defined(__APPLE__)
  char buffer[256];
  size_t size = sizeof(buffer);
  if (sysctlbyname("kern.osrelease", buffer, &size, NULL, 0) == 0) {
    ss << "macOS Kernel Version: " << buffer << "\n";
  }

  uint32_t count = _dyld_image_count();
  ss << "Executable Path: " << _dyld_get_image_name(0) << "\n";

#else
  struct utsname uts;
  if (uname(&uts) == 0) {
    ss << "System: " << uts.sysname << "\n"
       << "Release: " << uts.release << "\n"
       << "Version: " << uts.version << "\n"
       << "Machine: " << uts.machine << "\n";
  }
#endif

  return ss.str();
}

std::string getStackTrace() {
  std::stringstream ss;

#if defined(_WIN32)
  HANDLE process = GetCurrentProcess();
  SymInitialize(process, NULL, TRUE);

  void *stack[62];
  WORD frames = CaptureStackBackTrace(0, 62, stack, NULL);

  SYMBOL_INFO *symbol =
      (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
  symbol->MaxNameLen = 255;
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

  for (WORD i = 0; i < frames; i++) {
    SymFromAddr(process, (DWORD64)stack[i], 0, symbol);
    ss << i << ": " << symbol->Name << " at 0x" << std::hex << symbol->Address
       << std::dec << "\n";
  }

  free(symbol);
  SymCleanup(process);

#elif defined(__APPLE__) || defined(__linux__)
  void *array[50];
  int size = backtrace(array, 50);
  char **messages = backtrace_symbols(array, size);

  for (int i = 0; i < size && messages != NULL; i++) {
    Dl_info info;
    if (dladdr(array[i], &info)) {
      int status;
      char *demangled =
          abi::__cxa_demangle(info.dli_sname, NULL, NULL, &status);
      ss << i << ": "
         << (demangled        ? demangled
             : info.dli_sname ? info.dli_sname
                              : messages[i])
         << " at " << array[i] << "\n";
      free(demangled);
    } else {
      ss << i << ": " << messages[i] << "\n";
    }
  }
  free(messages);
#endif

  return ss.str();
}

} // namespace CrashHandler