#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace action_go::runtime {

class SharedMemoryDepthStream {
 public:
  SharedMemoryDepthStream(const std::string& name, size_t pixel_count, double timeout_sec);
  ~SharedMemoryDepthStream();
  std::vector<float> observation();
  uint64_t sequence() const;
  double age_sec() const;
  std::string stats() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}
