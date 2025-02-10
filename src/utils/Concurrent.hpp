#include <QFuture>
#include <QFutureWatcher>
#include <QPromise>
#include <QtConcurrent/QtConcurrent>
#include <QThreadPool>
#include <QTimer>
#include <QCoreApplication>
#include <coroutine>
#include <exception>
#include <type_traits>
#include <memory>
#include <vector>

namespace QtEx {

// -------------------- QFutureAwaiter & operator co_await --------------------
template<typename T>
struct QFutureAwaiter {
    // Explicit constructor to avoid copy-init issues
    explicit QFutureAwaiter(QFuture<T> f)
        : future(std::move(f)) {}

    QFuture<T> future;
    QFutureWatcher<T> watcher;

    bool await_ready() const { return future.isFinished(); }

    void await_suspend(std::coroutine_handle<> h) {
        QObject::connect(&watcher, &QFutureWatcher<T>::finished,
                         [h]() { h.resume(); });
        watcher.setFuture(future);
    }

    T await_resume() {
        if (future.isCanceled())
            throw std::runtime_error("Operation canceled");
        return future.result();
    }
};

template<typename T>
QFutureAwaiter<T> operator co_await(QFuture<T> f) {
    // Return explicit awaiter object
    return QFutureAwaiter<T>(std::move(f));
}

// -------------------- AsyncTask --------------------
template <typename T> 
class [[nodiscard]] AsyncTask {
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

    explicit AsyncTask(std::coroutine_handle<promise_type> h, QFuture<T> f)
        : handle(h), future(std::move(f)) {}

    ~AsyncTask() {
        if (handle)
            handle.destroy();
    }

    // Delegate actual suspend/resume to QFutureAwaiter
    bool await_ready() const noexcept { return future.isFinished(); }

    template <typename U>
    void await_suspend(std::coroutine_handle<U> h) {
        QObject::connect(&wrapper, &QFutureWatcher<T>::finished,
                         h.promise().resume_callback);
        wrapper.setFuture(future);
    }

    T await_resume() {
        if (wrapper.future().isCanceled())
            throw std::runtime_error("Operation canceled");
        if (auto ex = handle.promise().exception)
            std::rethrow_exception(ex);
        return future.result();
    }

    QFuture<T> getFuture() const { return future; }

private:
    std::coroutine_handle<promise_type> handle;
    QFuture<T> future;
    QFutureWatcher<T> wrapper;
};

template <> struct AsyncTask<void> {
public:
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
        : handle(h), future(std::move(f)) {}

    ~AsyncTask() {
        if (handle)
            handle.destroy();
    }

    bool await_ready() const noexcept { return future.isFinished(); }

    template <typename U>
    void await_suspend(std::coroutine_handle<U> h) {
        QObject::connect(&wrapper, &QFutureWatcher<void>::finished,
                         h.promise().resume_callback);
        wrapper.setFuture(future);
    }

    void await_resume() {
        if (wrapper.future().isCanceled())
            throw std::runtime_error("Operation canceled");
        if (auto ex = handle.promise().exception)
            std::rethrow_exception(ex);
        future.waitForFinished();
    }

    QFuture<void> getFuture() const { return future; }

private:
    std::coroutine_handle<promise_type> handle;
    QFuture<void> future;
    QFutureWatcher<void> wrapper;
};

// -------------------- Concurrent --------------------
class Concurrent {
public:
    template <typename F, typename... Args>
    static auto run(F&& func, Args&&... args,
                    QThreadPool* pool = QThreadPool::globalInstance())
        -> QFuture<std::invoke_result_t<F, Args...>>
    {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto promise = std::make_shared<QPromise<ReturnType>>();
        auto fut = promise->future();

        pool->start([promise, fun = std::forward<F>(func),
                    ...as = std::forward<Args>(args)]() mutable {
            try {
                if constexpr (std::is_void_v<ReturnType>) {
                    std::invoke(fun, as...);
                    promise->addResult();
                } else {
                    promise->addResult(std::invoke(fun, as...));
                }
                promise->finish();
            } catch (...) {
                promise->setException(std::current_exception());
            }
        });

        return fut;
    }

    // Simplify mappedReduced so it doesn't rely on .then() or .onFailed()
    template <typename Iterator, typename MapFunctor, typename ReduceFunctor>
    static auto mappedReduced(
        QThreadPool* pool, Iterator begin, Iterator end, MapFunctor map,
        ReduceFunctor reduce,
        QtConcurrent::ReduceOptions options = QtConcurrent::UnorderedReduce)
    {
        using ValueType = typename std::iterator_traits<Iterator>::value_type;
        using MapResult = std::invoke_result_t<MapFunctor, ValueType>;
        // For a simplistic approach, treat reduce's final type the same as map's return:
        using ReduceResult = MapResult;
        return QtConcurrent::mappedReduced<MapResult>(
            begin, end,
            [map](const ValueType& item) { return map(item); },
            [reduce](ReduceResult &acc, const MapResult &val) {
                reduce(acc, val);
            },
            options);
    }

    template <typename F, typename... Args>
    static AsyncTask<std::invoke_result_t<F, Args...>> async(F&& func,
                                                             Args&&... args)
    {
        auto future = run(std::forward<F>(func), std::forward<Args>(args)...);
        // Note: we rely on operator co_await(QFuture<T>), not a raw co_await on QFuture<T>
        co_return co_await future;
    }

    template <typename F, typename... Args>
    static QFuture<std::invoke_result_t<F, Args...>>
    runWithCancel(std::stop_token st, QThreadPool* pool, F&& func,
                  Args&&... args)
    {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto promise = std::make_shared<QPromise<ReturnType>>();
        auto fut = promise->future();

        pool->start([promise, st, f = std::forward<F>(func),
                    ...as = std::forward<Args>(args)]() mutable {
            try {
                if (st.stop_requested())
                    throw std::runtime_error("Operation canceled");
                if constexpr (std::is_void_v<ReturnType>) {
                    std::invoke(f, as...);
                    promise->addResult();
                } else {
                    promise->addResult(std::invoke(f, as...));
                }
                promise->finish();
            } catch (...) {
                promise->setException(std::current_exception());
            }
        });

        return fut;
    }
};

// -------------------- ExceptionAwareFuture --------------------
template <typename T>
class ExceptionAwareFuture : public QFuture<T> {
public:
    using Base = QFuture<T>;
    ExceptionAwareFuture(QFuture<T>&& future)
        : Base(std::move(future))
    {}

    template <typename Handler>
    ExceptionAwareFuture& onException(Handler&& handler) {
        auto* watcher = new QFutureWatcher<T>;
        QObject::connect(watcher, &QFutureWatcher<T>::finished, [=]() {
            try {
                watcher->future().result();
            } catch (...) {
                handler(std::current_exception());
            }
            watcher->deleteLater();
        });
        watcher->setFuture(*this);
        return *this;
    }

    template <typename ThenHandler, typename ErrHandler>
    ExceptionAwareFuture& then(ThenHandler&& thenHandler,
                               ErrHandler&& errHandler)
    {
        auto* watcher = new QFutureWatcher<T>;
        QObject::connect(watcher, &QFutureWatcher<T>::finished, [=]() {
            try {
                T r = watcher->result();
                thenHandler(r);
            } catch (...) {
                errHandler(std::current_exception());
            }
            watcher->deleteLater();
        });
        watcher->setFuture(*this);
        return *this;
    }

    T get() const {
        return this->Base::result();
    }
};

template <typename F, typename... Args>
auto async(F&& func, Args&&... args) {
    return Concurrent::async(std::forward<F>(func), std::forward<Args>(args)...);
}

template <typename T>
ExceptionAwareFuture<T> make_exception_aware(QFuture<T> future) {
    return ExceptionAwareFuture<T>(std::move(future));
}

// -------------------- Example coroutine --------------------
AsyncTask<QString> coroutineExample() {
    // Now co_await works correctly with QFuture, via QFutureAwaiter
    auto future = Concurrent::run([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return QString("协程任务完成");
    });
    auto result = co_await future;
    co_return result;
}

} // namespace QtEx

#include <QDebug>
#include <QString>
#include <chrono>
#include <thread>
#include <stop_token>

// -------------------- 示例函数 --------------------
void basicExample() {
    auto longOperation = []() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return QString("操作完成");
    };
    auto future = QtEx::Concurrent::run(longOperation);
    future.then([](const QString& result) {
        qDebug() << "任务结果:" << result;
    });
}

QtEx::AsyncTask<QString> coroutineExample() {
    auto future = QtEx::Concurrent::run([]() {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return QString("协程任务完成");
    });
    auto result = co_await future;
    co_return result;
}

void exceptionExample() {
    auto riskyOperation = []() -> QString {
        throw std::runtime_error("发生错误");
    };
    auto future = QtEx::make_exception_aware(
                      QtEx::Concurrent::run(riskyOperation));
    future.then(
        [](const QString& result) {
            qDebug() << "成功:" << result;
        },
        [](const std::exception_ptr& e) {
            try {
                std::rethrow_exception(e);
            } catch(const std::exception& ex) {
                qDebug() << "捕获到异常:" << ex.what();
            }
        }
    );
}

void mapReduceExample() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    auto future = QtEx::Concurrent::mappedReduced(
        QThreadPool::globalInstance(),
        numbers.begin(), numbers.end(),
        [](int n) { return n * n; },
        [](int& result, const int& value) { result += value; }
    );
    future.then([](int total) {
        qDebug() << "平方和:" << total;
    });
}

void cancellableExample() {
    std::stop_source ss;
    auto future = QtEx::Concurrent::runWithCancel(
        ss.get_token(),
        QThreadPool::globalInstance(),
        []() {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            return QString("完成");
        }
    );
    QTimer::singleShot(1000, [&ss]() { ss.request_stop(); });
    QtEx::make_exception_aware(future).then(
        [](const QString& result) {
            qDebug() << "结果:" << result;
        },
        [](const std::exception_ptr& e) {
            try {
                std::rethrow_exception(e);
            } catch(const std::exception& ex) {
                qDebug() << "任务被取消:" << ex.what();
            }
        }
    );
}

// -------------------- main --------------------
int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    basicExample();
    auto coroTask = coroutineExample();
    coroTask.getFuture().then([](const QString& result) {
        qDebug() << result;
    });
    exceptionExample();
    mapReduceExample();
    cancellableExample();

    return app.exec();
}