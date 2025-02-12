#pragma once

#include <QtCore/QObject>
#include <QtCore/QString>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

class DynamicThreadPool : public QObject {
  Q_OBJECT

public:
  enum class Priority { High, Normal, Low };

  explicit DynamicThreadPool(
      int min_threads = std::thread::hardware_concurrency(),
      int max_threads = 4 * std::thread::hardware_concurrency(),
      std::chrono::milliseconds idle_timeout = std::chrono::seconds(30),
      QObject *parent = nullptr);

  ~DynamicThreadPool();

  template <typename F, typename... Args>
  auto enqueue(F &&func, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  template <typename F, typename... Args>
  auto enqueueWithPriority(Priority priority, F &&func, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  struct TaskHandle {
    std::shared_ptr<std::atomic<bool>> cancelled =
        std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::promise<void>> cancel_promise;

    // 添加移动构造和赋值
    TaskHandle() = default;
    TaskHandle(TaskHandle &&) noexcept = default;
    TaskHandle &operator=(TaskHandle &&) noexcept = default;
    // 删除复制
    TaskHandle(const TaskHandle &) = delete;
    TaskHandle &operator=(const TaskHandle &) = delete;
  };

  template <typename F, typename... Args>
  std::pair<TaskHandle, std::future<std::invoke_result_t<F, Args...>>>
  enqueueCancelable(F &&func, Args &&...args);

  void waitAll();
  void cancelAll();

  // 配置修改方法
  void setMinThreads(int min);
  void setMaxThreads(int max);
  void setIdleTimeout(std::chrono::milliseconds timeout);

  // 状态查询
  int activeThreads() const { return active_workers_.load(); }
  int pendingTasks() const {
    std::shared_lock lock(queue_mutex_);
    return task_queue_.size() + delayed_queue_.size();
  }
  int currentThreadCount() const { return current_threads_.load(); }

signals:
  void taskStarted(quint64 taskId);
  void taskFinished(quint64 taskId);
  void taskFailed(quint64 taskId, const QString &error);
  void taskCancelled(quint64 taskId);
  void threadCountChanged(int count);

private:
  struct PrioritizedTask {
    std::packaged_task<void()> task;
    Priority priority;
    quint64 task_id;
    std::chrono::steady_clock::time_point enqueue_time;
    std::weak_ptr<TaskHandle> handle;

    PrioritizedTask(std::packaged_task<void()> &&t, Priority p, quint64 id,
                    std::chrono::steady_clock::time_point time,
                    std::weak_ptr<TaskHandle> h)
        : task(std::move(t)), priority(p), task_id(id), enqueue_time(time),
          handle(std::move(h)) {}

    PrioritizedTask(PrioritizedTask &&other) noexcept = default;
    PrioritizedTask &operator=(PrioritizedTask &&other) noexcept = default;

    // 删除复制
    PrioritizedTask(const PrioritizedTask &) = delete;
    PrioritizedTask &operator=(const PrioritizedTask &) = delete;

    bool operator<(const PrioritizedTask &other) const {
      if (priority == other.priority) {
        return enqueue_time > other.enqueue_time; // 时间早的优先
      }
      return priority < other.priority;
    }
  };

  void workerRoutine(std::stop_token st);
  void managerRoutine();
  void expandIfNeeded();
  void shrinkIfPossible();
  void processDelayedTasks();
  bool stealTask(PrioritizedTask &stolen_task);

  // 配置参数（改为atomic支持动态修改）
  std::atomic<int> min_threads_;
  std::atomic<int> max_threads_;
  std::atomic<std::chrono::milliseconds> idle_timeout_;

  // 同步原语
  mutable std::shared_mutex queue_mutex_;
  std::condition_variable_any queue_cv_;
  std::mutex control_mutex_;
  std::mutex delay_mutex_;

  // 状态管理
  std::atomic<int> current_threads_{0};
  std::atomic<int> active_workers_{0};
  std::atomic<bool> shutdown_{false};
  std::atomic<quint64> next_task_id_{1};

  // 任务队列
  std::priority_queue<PrioritizedTask> task_queue_;
  std::unordered_map<quint64, PrioritizedTask> delayed_queue_;
  std::vector<std::jthread> workers_;
  std::jthread manager_thread_;

  // 工作窃取相关
  static constexpr size_t WORK_STEAL_ATTEMPTS = 2;
  std::atomic<size_t> steal_index_{0};
};

template <typename F, typename... Args>
auto DynamicThreadPool::enqueue(F &&func, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  return enqueueWithPriority(Priority::Normal, std::forward<F>(func),
                             std::forward<Args>(args)...);
}

template <typename F, typename... Args>
auto DynamicThreadPool::enqueueWithPriority(Priority priority, F &&func,
                                            Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  auto task = std::packaged_task<ReturnType()>(
      std::bind(std::forward<F>(func), std::forward<Args>(args)...));
  auto future = task.get_future();
  const auto task_id = next_task_id_.fetch_add(1);

  {
    std::unique_lock lock(queue_mutex_);
    task_queue_.push(PrioritizedTask(
        std::packaged_task<void()>(std::move(task)), priority, task_id,
        std::chrono::steady_clock::now(), std::weak_ptr<TaskHandle>()));
  }

  QMetaObject::invokeMethod(
      this, [this, task_id] { emit taskStarted(task_id); },
      Qt::QueuedConnection);
  queue_cv_.notify_one();
  return future;
}

template <typename F, typename... Args>
std::pair<DynamicThreadPool::TaskHandle,
          std::future<std::invoke_result_t<F, Args...>>>
DynamicThreadPool::enqueueCancelable(F &&func, Args &&...args) {
  using ReturnType = std::invoke_result_t<F, Args...>;

  TaskHandle handle;
  auto promise = std::make_shared<std::promise<void>>();
  handle.cancel_promise = promise;

  auto task = std::packaged_task<ReturnType()>(
      [func = std::forward<F>(func),
       args = std::make_tuple(std::forward<Args>(args)...),
       handle = std::move(handle)]() mutable {
        if (handle.cancelled->load()) {
          throw std::runtime_error("Task cancelled");
        }
        return std::apply(func, args);
      });

  auto future = task.get_future();
  const auto task_id = next_task_id_.fetch_add(1);

  {
    std::unique_lock lock(queue_mutex_);
    task_queue_.push(PrioritizedTask(
        std::packaged_task<void()>(std::move(task)), Priority::Normal, task_id,
        std::chrono::steady_clock::now(),
        std::weak_ptr<TaskHandle>(std::make_shared<TaskHandle>(handle))));
  }

  QMetaObject::invokeMethod(
      this, [this, task_id] { emit taskStarted(task_id); },
      Qt::QueuedConnection);
  queue_cv_.notify_one();
  return {handle, std::move(future)};
}
