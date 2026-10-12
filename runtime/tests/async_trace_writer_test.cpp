#include "action_go/runtime/async_trace_writer.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <unistd.h>

int main() {
  const auto path = std::filesystem::temp_directory_path() /
      ("action_go_async_trace_" + std::to_string(getpid()) + ".jsonl");
  const auto caller = std::this_thread::get_id();
  std::promise<void> entered;
  std::promise<void> release;
  auto released = release.get_future().share();
  {
    action_go::runtime::AsyncTraceWriter writer(path.string());
    writer.submit([&] {
      assert(std::this_thread::get_id() != caller);
      entered.set_value();
      released.wait();
      return std::string("0");
    });
    entered.get_future().wait();
    for (int index = 1; index < 100; ++index) {
      writer.submit([index, caller] {
        assert(std::this_thread::get_id() != caller);
        return std::to_string(index);
      });
    }
    release.set_value();
  }
  std::ifstream input(path);
  std::string line;
  for (int index = 0; index < 100; ++index) {
    assert(std::getline(input, line));
    assert(line == std::to_string(index));
  }
  assert(!std::getline(input, line));
  std::filesystem::remove(path);
}
