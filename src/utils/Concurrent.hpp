#include <QFuture>
#include <QFutureWatcher>
#include <QPromise>
#include <QtConcurrent/QtConcurrent>
#include <coroutine>
#include <exception>
#include <type_traits>

namespace QtEx {

template <typename T> class [[nodiscard]] AsyncTask {
public:
  struct promise_type {
    QPromise<T> promise;
    QFutureWatcher<T> watcher;
    std::exception_ptr exception;

    AsyncTask<T> get_return_object() {
      return AsyncTask<T>{
          std::coroutine_handle<promise_type>::from_promise(*this),
          promise.future()};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_value(T value) { promise.addResult(std::move(value)); }
    void unhandled_exception() {
      exception = std::current_exception();
      promise.setException(exception);
    }
  };

  // 修改 handle 类型为 std::coroutine_handle<promise_type>
  explicit AsyncTask(std::coroutine_handle<promise_type> h, QFuture<T> f)
      : handle(h), future(f) {}

  ~AsyncTask() {
    if (handle)
      handle.destroy();
  }

  bool await_ready() const noexcept { return future.isFinished(); }

  template <typename U> void await_suspend(std::coroutine_handle<U> h) {
    QObject::connect(&futureWatcher, &QFutureWatcher<T>::finished,
                     h.promise().resume_callback);
    futureWatcher.setFuture(future);
  }

  T await_resume() {
    if (futureWatcher.future().isCanceled()) {
      throw std::runtime_error("Operation canceled");
    }
    // 修正 handle 的访问方式
    if (auto ex = handle.promise().exception) {
      std::rethrow_exception(ex);
    }
    return future.result();
  }

  QFuture<T> getFuture() const { return future; }

private:
  std::coroutine_handle<promise_type> handle; // 修改后的类型
  QFuture<T> future;
  QFutureWatcher<T> futureWatcher;
};

template <> struct AsyncTask<void> {
  struct promise_type {
    QPromise<void> promise;
    QFutureWatcher<void> watcher;
    std::exception_ptr exception;

    AsyncTask<void> get_return_object() {
      return AsyncTask<void>{
          std::coroutine_handle<promise_type>::from_promise(*this),
          promise.future()};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }

    void return_void() { promise.finish(); }

    void unhandled_exception() {
      exception = std::current_exception();
      promise.setException(exception);
    }
  };

  explicit AsyncTask(std::coroutine_handle<promise_type> h, QFuture<void> f)
      : handle(h), future(f) {}

  ~AsyncTask() {
    if (handle)
      handle.destroy();
  }

  bool await_ready() const noexcept { return future.isFinished(); }

  template <typename U> void await_suspend(std::coroutine_handle<U> h) {
    QObject::connect(&futureWatcher, &QFutureWatcher<void>::finished,
                     h.promise().resume_callback);
    futureWatcher.setFuture(future);
  }

  void await_resume() {
    if (futureWatcher.future().isCanceled()) {
      throw std::runtime_error("Operation canceled");
    }
    if (auto ex = handle.promise().exception) {
      std::rethrow_exception(ex);
    }
    future.waitForFinished();
  }

  QFuture<void> getFuture() const { return future; }

private:
  std::coroutine_handle<promise_type> handle;
  QFuture<void> future;
  QFutureWatcher<void> futureWatcher;
};

// 核心执行器封装
class Concurrent {
public:
  template <typename F, typename... Args>
  static auto run(F &&func, Args &&...args,
                  QThreadPool *pool = QThreadPool::globalInstance())
      -> QFuture<std::invoke_result_t<F, Args...>> {
    using ReturnType = std::invoke_result_t<F, Args...>;

    QPromise<ReturnType> promise;
    auto future = promise.future();

    QtConcurrent::run(pool, [=, func = std::forward<F>(func)]() mutable {
      try {
        if constexpr (std::is_void_v<ReturnType>) {
          std::invoke(func, args...);
          promise.addResult();
        } else {
          promise.addResult(std::invoke(func, args...));
        }
        promise.finish();
      } catch (...) {
        promise.setException(std::current_exception());
      }
    });

    return future;
  }

  template <typename Iterator, typename MapFunctor, typename ReduceFunctor>
  static auto mappedReduced(
      QThreadPool *pool, Iterator begin, Iterator end, MapFunctor map,
      ReduceFunctor reduce,
      QtConcurrent::ReduceOptions options = QtConcurrent::UnorderedReduce)
      -> QFuture<decltype(map(*begin))> {
    using MapResult = decltype(map(*begin));
    using ReduceResult = decltype(reduce(MapResult{}, MapResult{}));

    QPromise<ReduceResult> promise;
    auto future = promise.future();

    // 修改 QtConcurrent::mapReduce 为 QtConcurrent::mappedReduced
    QtConcurrent::mappedReduced(
        pool, begin, end,
        [=](typename std::iterator_traits<Iterator>::value_type item) {
          try {
            return map(item);
          } catch (...) {
            promise.setException(std::current_exception());
            throw;
          }
        },
        [=](ReduceResult &result, const MapResult &value) {
          try {
            reduce(result, value);
          } catch (...) {
            promise.setException(std::current_exception());
            throw;
          }
        },
        options)
        .then([=]() mutable { promise.finish(); })
        .onFailed([=] { promise.setException(std::current_exception()); });

    return future;
  }

  // 协程适配器
  template <typename F, typename... Args>
  static AsyncTask<std::invoke_result_t<F, Args...>> async(F &&func,
                                                           Args &&...args) {
    auto future = run(QThreadPool::globalInstance(), std::forward<F>(func),
                      std::forward<Args>(args)...);
    co_return co_await future;
  }

  // 带取消功能的执行器
  template <typename F, typename... Args>
  static QFuture<std::invoke_result_t<F, Args...>>
  runWithCancel(std::stop_token st, QThreadPool *pool, F &&func,
                Args &&...args) {
    using ReturnType = std::invoke_result_t<F, Args...>;

    QPromise<ReturnType> promise;
    auto future = promise.future();

    QtConcurrent::run(pool, [=, func = std::forward<F>(func)]() mutable {
      try {
        if (st.stop_requested()) {
          throw std::runtime_error("Operation canceled");
        }

        if constexpr (std::is_void_v<ReturnType>) {
          std::invoke(func, args...);
          promise.addResult();
        } else {
          promise.addResult(std::invoke(func, args...));
        }
        promise.finish();
      } catch (...) {
        promise.setException(std::current_exception());
      }
    });

    return future;
  }
};

// 异常感知Future扩展
template <typename T> class ExceptionAwareFuture : public QFuture<T> {
public:
  using Base = QFuture<T>;

  ExceptionAwareFuture(QFuture<T> &&future) : Base(std::move(future)) {}

  template <typename Handler>
  ExceptionAwareFuture &onException(Handler &&handler) {
    QFutureWatcher<T> *watcher = new QFutureWatcher<T>;
    QObject::connect(watcher, &QFutureWatcher<T>::finished, [=] {
      if (watcher->future().isCanceled()) {
        handler(std::make_exception_ptr(std::runtime_error("Canceled")));
      } else if (watcher->future().exceptionCount() > 0) {
        handler(watcher->future().exception());
      }
      watcher->deleteLater();
    });
    watcher->setFuture(*this);
    return *this;
  }

  template <typename ThenHandler, typename ErrHandler>
  ExceptionAwareFuture &then(ThenHandler &&thenHandler,
                             ErrHandler &&errHandler) {
    QFutureWatcher<T> *watcher = new QFutureWatcher<T>;
    QObject::connect(watcher, &QFutureWatcher<T>::finished, [=] {
      if (watcher->future().exceptionCount() > 0) {
        errHandler(watcher->future().exception());
      } else {
        if constexpr (std::is_invocable_v<ThenHandler, T>) {
          thenHandler(watcher->result());
        } else {
          thenHandler();
        }
      }
      watcher->deleteLater();
    });
    watcher->setFuture(*this);
    return *this;
  }

  T get() const {
    if (this->isCanceled()) {
      throw std::runtime_error("Operation canceled");
    }
    if (this->exceptionCount() > 0) {
      std::rethrow_exception(this->exception());
    }
    return this->result();
  }
};

// 自动推导辅助函数
template <typename F, typename... Args> auto async(F &&func, Args &&...args) {
  return Concurrent::async(std::forward<F>(func), std::forward<Args>(args)...);
}

template <typename T>
ExceptionAwareFuture<T> make_exception_aware(QFuture<T> future) {
  return ExceptionAwareFuture<T>(std::move(future));
}

} // namespace QtEx