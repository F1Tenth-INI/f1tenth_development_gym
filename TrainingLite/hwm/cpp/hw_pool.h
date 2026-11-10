// Persistent worker pool: ``run(n, fn)`` calls fn(0) .. fn(n - 1) on the workers and the calling thread.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace hw {

class Pool {
public:
    ~Pool() { resize(1); }

    // Total threads including the caller. Not thread safe against a concurrent run().
    void resize(int threads) {
        threads = threads < 1 ? 1 : threads;
        if ((int)workers_.size() == threads - 1) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        wake_.notify_all();
        for (auto& w : workers_) w.join();
        workers_.clear();
        stop_ = false;
        for (int i = 0; i < threads - 1; ++i) workers_.emplace_back([this] { loop(); });
    }

    int threads() const { return (int)workers_.size() + 1; }

    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        task_ = &fn;
        count_ = n;
        next_.store(0);
        busy_ = (int)workers_.size();
        ++generation_;
        lock.unlock();
        wake_.notify_all();
        work();
        lock.lock();
        done_.wait(lock, [this] { return busy_ == 0; });
        task_ = nullptr;
    }

private:
    void work() {
        for (int i = next_.fetch_add(1); i < count_; i = next_.fetch_add(1)) (*task_)(i);
    }

    void loop() {
        unsigned long seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_ != seen; });
            seen = generation_;
            if (stop_) return;
            lock.unlock();
            work();
            lock.lock();
            if (--busy_ == 0) done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_, done_;
    const std::function<void(int)>* task_ = nullptr;
    std::atomic<int> next_{0};
    int count_ = 0, busy_ = 0;
    unsigned long generation_ = 0;
    bool stop_ = false;
};

Pool& pool();

}  // namespace hw
