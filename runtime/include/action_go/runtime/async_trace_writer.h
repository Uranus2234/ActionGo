#pragma once

#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace action_go::runtime {

class AsyncTraceWriter {
 public:
  using Record = std::function<std::string()>;

  explicit AsyncTraceWriter(const std::string& path) : stream_(path), path_(path) {
    if (!stream_) throw std::runtime_error("cannot open trace: " + path);
    worker_ = std::thread(&AsyncTraceWriter::write_loop, this);
  }

  ~AsyncTraceWriter() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_one();
    worker_.join();
    if (dropped_ > 0) {
      std::cerr << "[trace] incomplete: " << dropped_ << " records dropped from " << path_ << '\n';
    }
  }

  void submit(Record record) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.size() >= 256 || failed_) {
        ++dropped_;
        return;
      }
      queue_.push_back(std::move(record));
    }
    condition_.notify_one();
  }

  AsyncTraceWriter(const AsyncTraceWriter&) = delete;
  AsyncTraceWriter& operator=(const AsyncTraceWriter&) = delete;

 private:
  void write_loop() {
    while (true) {
      Record record;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) return;
        record = std::move(queue_.front());
        queue_.pop_front();
      }
      try {
        stream_ << record() << '\n';
        stream_.flush();
        if (!stream_) throw std::runtime_error("trace write failed");
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        dropped_ += queue_.size() + 1;
        queue_.clear();
        std::cerr << "[trace] " << path_ << ": " << error.what() << '\n';
      }
    }
  }

  std::ofstream stream_;
  std::string path_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Record> queue_;
  bool stopping_{false};
  bool failed_{false};
  size_t dropped_{0};
  std::thread worker_;
};

}
