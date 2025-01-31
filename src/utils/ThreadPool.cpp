#include "ThreadPool.hpp"

// 实现部分
DynamicThreadPool::DynamicThreadPool(int min_threads, int max_threads,
                                     std::chrono::milliseconds idle_timeout,
                                     QObject *parent)
    : QObject(parent), min_threads_(std::max(1, min_threads)),
      max_threads_(std::max(min_threads, max_threads)),
      idle_timeout_(idle_timeout) {
  for (int i = 0; i < min_threads_; ++i) {
    workers_.emplace_back([this](std::stop_token st) { workerRoutine(st); });
    ++current_threads_;
  }
  manager_thread_ =
      std::jthread([this](std::stop_token st) { managerRoutine(); });
}

DynamicThreadPool::~DynamicThreadPool() {
  shutdown_.store(true);
  queue_cv_.notify_all();
  manager_thread_.request_stop();
}

void DynamicThreadPool::workerRoutine(std::stop_token st) {
  while (!st.stop_requested()) {
    std::optional<PrioritizedTask> task;
    bool task_acquired = false;

    {
      std::unique_lock lock(queue_mutex_);
      ++active_workers_;

      const auto predicate = [this] {
        return !task_queue_.empty() || shutdown_.load();
      };

      if (queue_cv_.wait_for(lock, idle_timeout_.load(), predicate)) {
        if (!task_queue_.empty()) {
          auto tmpTop =
              std::move(const_cast<PrioritizedTask &>(task_queue_.top()));
          task_queue_.pop();
          task.emplace(std::move(tmpTop));
          task_acquired = true;
        }
      } else {
        // 超时后尝试工作窃取
        for (size_t i = 0; i < WORK_STEAL_ATTEMPTS; ++i) {
          if (stealTask(*task)) {
            task_acquired = true;
            break;
          }
        }
      }

      --active_workers_;
    }

    if (task_acquired && task) {
      try {
        // 检查任务是否被取消
        if (auto handle = task->handle.lock()) {
          if (handle->cancelled->load()) {
            QMetaObject::invokeMethod(
                this,
                [this, task_id = task->task_id] {
                  emit taskCancelled(task_id);
                },
                Qt::QueuedConnection);
            continue;
          }
        }

        task->task();
        QMetaObject::invokeMethod(
            this,
            [this, task_id = task->task_id] { emit taskFinished(task_id); },
            Qt::QueuedConnection);
      } catch (const std::exception &e) {
        const QString error = QString::fromUtf8(e.what());
        QMetaObject::invokeMethod(
            this,
            [this, task_id = task->task_id, error] {
              emit taskFailed(task_id, error);
            },
            Qt::QueuedConnection);
      }
    } else {
      shrinkIfPossible();
    }
  }
}

void DynamicThreadPool::managerRoutine() {
  while (!shutdown_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 处理延迟任务
    processDelayedTasks();

    // 动态调整线程池
    expandIfNeeded();
    shrinkIfPossible();
  }
}

void DynamicThreadPool::expandIfNeeded() {
  std::lock_guard lock(control_mutex_);
  const int current = current_threads_.load();
  const int pending = pendingTasks();

  if (current < max_threads_.load() && pending > current * 2) {
    workers_.emplace_back([this](std::stop_token st) { workerRoutine(st); });
    ++current_threads_;
    QMetaObject::invokeMethod(
        this, [this] { emit threadCountChanged(current_threads_.load()); },
        Qt::QueuedConnection);
  }
}

void DynamicThreadPool::shrinkIfPossible() {
  std::lock_guard lock(control_mutex_);
  const int current = current_threads_.load();
  const int min = min_threads_.load();

  if (current > min && active_workers_.load() < current * 0.3) {
    if (auto it = std::ranges::find_if(
            workers_,
            [](auto &t) { return t.get_stop_token().stop_requested(); });
        it != workers_.end()) {
      it->request_stop();
      it->detach();
      workers_.erase(it);
      --current_threads_;
      QMetaObject::invokeMethod(
          this, [this] { emit threadCountChanged(current_threads_.load()); },
          Qt::QueuedConnection);
    }
  }
}

bool DynamicThreadPool::stealTask(PrioritizedTask &stolen_task) {
  const size_t num_workers = workers_.size();
  if (num_workers < 2)
    return false;

  for (size_t i = 0; i < num_workers; ++i) {
    const size_t index = (steal_index_.fetch_add(1) + i) % num_workers;
    auto &worker = workers_[index];

    if (worker.get_id() == std::this_thread::get_id())
      continue;

    std::unique_lock lock(queue_mutex_, std::try_to_lock);
    if (!lock || task_queue_.empty())
      continue;

    auto tmp = std::move(const_cast<PrioritizedTask &>(task_queue_.top()));
    task_queue_.pop();
    stolen_task = std::move(tmp);
    return true;
  }
  return false;
}

void DynamicThreadPool::cancelAll() {
  std::unique_lock lock(queue_mutex_);
  while (!task_queue_.empty()) {
    auto tmpTask = std::move(const_cast<PrioritizedTask &>(task_queue_.top()));
    task_queue_.pop();
    if (auto handle = tmpTask.handle.lock()) {
      handle->cancelled->store(true);
      QMetaObject::invokeMethod(
          this,
          [this, task_id = tmpTask.task_id] { emit taskCancelled(task_id); },
          Qt::QueuedConnection);
    }
  }
}

void DynamicThreadPool::setMinThreads(int min) {
  min = std::max(1, min);
  min_threads_.store(min);
  if (current_threads_.load() < min) {
    expandIfNeeded();
  }
}

void DynamicThreadPool::setMaxThreads(int max) {
  max = std::max(min_threads_.load(), max);
  max_threads_.store(max);
  if (current_threads_.load() > max) {
    shrinkIfPossible();
  }
}

void DynamicThreadPool::setIdleTimeout(std::chrono::milliseconds timeout) {
  idle_timeout_.store(timeout);
}

void DynamicThreadPool::waitAll() {
  std::unique_lock lock(queue_mutex_);
  queue_cv_.wait(lock, [this] {
    return task_queue_.empty() && (active_workers_.load() == 0);
  });
}