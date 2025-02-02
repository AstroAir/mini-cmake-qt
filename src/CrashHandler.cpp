#include "CrashHandler.hpp"

#include <cpuid.h>
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
CrashHandler::Config g_config;
std::function<void(const std::string &)> g_customHandler;

#ifdef _WIN32
void createMiniDump(EXCEPTION_POINTERS *exceptionPointers) {
  HANDLE hFile = CreateFileA("crash.dmp", GENERIC_WRITE, FILE_SHARE_READ, 0,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
  if (hFile == INVALID_HANDLE_VALUE)
    return;

  MINIDUMP_EXCEPTION_INFORMATION exInfo;
  exInfo.ThreadId = GetCurrentThreadId();
  exInfo.ExceptionPointers = exceptionPointers;
  exInfo.ClientPointers = FALSE;

  MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hFile,
                    MiniDumpNormal, &exInfo, nullptr, nullptr);
  CloseHandle(hFile);
}
#endif

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

  createMiniDump(pExceptionInfo);

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

// 新增：获取CPU信息
std::string getCpuInfoImpl() {
  std::stringstream ss;
#ifdef _WIN32
  int cpuInfo[4] = {-1};
  __cpuid(cpuInfo, 0x80000000);
  unsigned int nExIds = cpuInfo[0];
  char brand[64] = {0};

  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(cpuInfo, i);
    if (i == 0x80000002)
      memcpy(brand, cpuInfo, sizeof(cpuInfo));
    else if (i == 0x80000003)
      memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));
    else if (i == 0x80000004)
      memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
  }
  ss << "CPU: " << brand;
#endif
  return ss.str();
}

// 新增：获取内存信息
std::string getMemoryInfoImpl() {
  std::stringstream ss;
#ifdef _WIN32
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  ss << "Total Physical Memory: " << memInfo.ullTotalPhys / (1024 * 1024)
     << " MB\n";
  ss << "Available Memory: " << memInfo.ullAvailPhys / (1024 * 1024) << " MB\n";
  ss << "Memory Load: " << memInfo.dwMemoryLoad << "%\n";
#endif
  return ss.str();
}

#endif

std::string getCpuInfoImpl() {
  std::stringstream ss;
#if defined(_WIN32)
  int cpuInfo[4] = {-1};
  char brand[64] = {0};

  __cpuidex(cpuInfo, 0x80000002, 0);
  memcpy(brand, cpuInfo, sizeof(cpuInfo));
  __cpuidex(cpuInfo, 0x80000003, 0);
  memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));
  __cpuidex(cpuInfo, 0x80000004, 0);
  memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));

  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);

  ss << "CPU Brand: " << brand << "\n"
     << "Number of Cores: " << sysInfo.dwNumberOfProcessors << "\n"
     << "Page Size: " << sysInfo.dwPageSize << " bytes\n";

#elif defined(__APPLE__)
  char buffer[1024];
  size_t size = sizeof(buffer);

  if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nullptr, 0) ==
      0) {
    ss << "CPU Brand: " << buffer << "\n";
  }

  if (sysctlbyname("hw.physicalcpu", &buffer, &size, nullptr, 0) == 0) {
    ss << "Physical CPUs: " << buffer << "\n";
  }

  if (sysctlbyname("hw.logicalcpu", &buffer, &size, nullptr, 0) == 0) {
    ss << "Logical CPUs: " << buffer << "\n";
  }

#else // Linux
  FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
  if (cpuinfo) {
    char line[1024];
    while (fgets(line, sizeof(line), cpuinfo)) {
      if (strncmp(line, "model name", 10) == 0) {
        char *model = strchr(line, ':');
        if (model) {
          ss << "CPU Model:" << (model + 2);
          break;
        }
      }
    }
    fclose(cpuinfo);
  }

  // 获取CPU核心数
  if (sysconf(_SC_NPROCESSORS_ONLN) != -1) {
    ss << "Number of CPUs: " << sysconf(_SC_NPROCESSORS_ONLN) << "\n";
  }
#endif
  return ss.str();
}

std::string getMemoryInfoImpl() {
  std::stringstream ss;
#if defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  if (GlobalMemoryStatusEx(&memInfo)) {
    ss << "Total Physical Memory: " << (memInfo.ullTotalPhys / (1024 * 1024))
       << " MB\n"
       << "Available Physical Memory: "
       << (memInfo.ullAvailPhys / (1024 * 1024)) << " MB\n"
       << "Memory Load: " << memInfo.dwMemoryLoad << "%\n"
       << "Total Virtual Memory: " << (memInfo.ullTotalVirtual / (1024 * 1024))
       << " MB\n"
       << "Available Virtual Memory: "
       << (memInfo.ullAvailVirtual / (1024 * 1024)) << " MB\n";
  }

#elif defined(__APPLE__)
  int mib[2];
  size_t size;
  struct vm_statistics64 vm_stats;

  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  uint64_t total_memory;
  size = sizeof(total_memory);
  if (sysctl(mib, 2, &total_memory, &size, NULL, 0) == 0) {
    ss << "Total Physical Memory: " << (total_memory / (1024 * 1024))
       << " MB\n";
  }

  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
    uint64_t free_memory = vm_stats.free_count * 4096;
    uint64_t used_memory = (vm_stats.active_count + vm_stats.inactive_count +
                            vm_stats.wire_count) *
                           4096;

    ss << "Free Memory: " << (free_memory / (1024 * 1024)) << " MB\n"
       << "Used Memory: " << (used_memory / (1024 * 1024)) << " MB\n";
  }

#else // Linux
  FILE *meminfo = fopen("/proc/meminfo", "r");
  if (meminfo) {
    char line[256];
    unsigned long total_mem = 0, free_mem = 0, buffers = 0, cached = 0;

    while (fgets(line, sizeof(line), meminfo)) {
      if (strncmp(line, "MemTotal:", 9) == 0)
        sscanf(line, "MemTotal: %lu", &total_mem);
      else if (strncmp(line, "MemFree:", 8) == 0)
        sscanf(line, "MemFree: %lu", &free_mem);
      else if (strncmp(line, "Buffers:", 8) == 0)
        sscanf(line, "Buffers: %lu", &buffers);
      else if (strncmp(line, "Cached:", 7) == 0)
        sscanf(line, "Cached: %lu", &cached);
    }
    fclose(meminfo);

    unsigned long used_mem = total_mem - free_mem - buffers - cached;
    ss << "Total Memory: " << (total_mem / 1024) << " MB\n"
       << "Used Memory: " << (used_mem / 1024) << " MB\n"
       << "Free Memory: " << (free_mem / 1024) << " MB\n"
       << "Buffers: " << (buffers / 1024) << " MB\n"
       << "Cached: " << (cached / 1024) << " MB\n";
  }
#endif
  return ss.str();
}

} // namespace

namespace CrashHandler {

void init(const Config &config) {
  g_config = config;
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

void setCustomHandler(std::function<void(const std::string &)> handler) {
  g_customHandler = handler;
}

std::string getMemoryInfo() { return getMemoryInfoImpl(); }

std::string getCpuInfo() { return getCpuInfoImpl(); }

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

std::string getStackTrace(size_t maxFrames) {
  std::stringstream ss;

#if defined(USE_BOOST_STACKTRACE)
  ss << boost::stacktrace::stacktrace();
#else
#if defined(_WIN32)
  HANDLE process = GetCurrentProcess();
  SymInitialize(process, NULL, TRUE);

  void *stack[256];
  WORD frames = CaptureStackBackTrace(
      0, (DWORD)std::min(maxFrames, (size_t)256), stack, NULL);

  SYMBOL_INFO *symbol =
      (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
  symbol->MaxNameLen = 255;
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

  IMAGEHLP_LINE64 line;
  line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
  DWORD displacement;

  for (WORD i = 0; i < frames; i++) {
    DWORD64 address = (DWORD64)stack[i];
    SymFromAddr(process, address, 0, symbol);

    ss << std::dec << i << ": " << symbol->Name;

    // 尝试获取文件和行号信息
    if (SymGetLineFromAddr64(process, address, &displacement, &line)) {
      ss << " at " << line.FileName << ":" << line.LineNumber;
    }

    ss << " [0x" << std::hex << symbol->Address << "]\n";
  }

  free(symbol);
  SymCleanup(process);

#elif defined(__APPLE__) || defined(__linux__)
  void *array[256];
  int size = backtrace(array, std::min(maxFrames, (size_t)256));
  char **messages = backtrace_symbols(array, size);

  for (int i = 0; i < size && messages != NULL; i++) {
    Dl_info info;
    if (dladdr(array[i], &info)) {
      int status;
      char *demangled =
          abi::__cxa_demangle(info.dli_sname, NULL, NULL, &status);

      ss << i << ": ";
      if (demangled) {
        ss << demangled << " in ";
        free(demangled);
      } else if (info.dli_sname) {
        ss << info.dli_sname << " in ";
      }

      if (info.dli_fname) {
        ss << info.dli_fname;
      }

      ss << " at " << array[i] << "\n";
    } else {
      ss << i << ": " << messages[i] << "\n";
    }
  }
  free(messages);
#endif
#endif

  return ss.str();
}

} // namespace CrashHandler