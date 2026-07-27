#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace action_go::runtime::npy {

struct ArrayView {
  std::string descr;
  std::vector<size_t> shape;
  const uint8_t* data{nullptr};
  size_t data_bytes{0};
};

inline uint16_t little_u16(const uint8_t* data) {
  return static_cast<uint16_t>(data[0]) |
         (static_cast<uint16_t>(data[1]) << 8);
}

inline uint32_t little_u32(const uint8_t* data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

inline std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline size_t element_count(const std::vector<size_t>& shape) {
  size_t count = 1;
  for (size_t dimension : shape) {
    if (dimension != 0 && count > std::numeric_limits<size_t>::max() / dimension) {
      throw std::runtime_error("npy shape element count overflows size_t");
    }
    count *= dimension;
  }
  return count;
}

inline ArrayView parse_header(const std::vector<uint8_t>& bytes) {
  if (bytes.size() < 10 || bytes[0] != 0x93 ||
      std::memcmp(bytes.data() + 1, "NUMPY", 5) != 0) {
    throw std::runtime_error("invalid npy header");
  }

  const uint8_t major = bytes[6];
  size_t header_start = 0;
  size_t header_length = 0;
  if (major == 1) {
    header_start = 10;
    header_length = little_u16(bytes.data() + 8);
  } else if (major == 2 || major == 3) {
    if (bytes.size() < 12) throw std::runtime_error("truncated npy v2/v3 header");
    header_start = 12;
    header_length = little_u32(bytes.data() + 8);
  } else {
    throw std::runtime_error("unsupported npy version");
  }
  if (header_length > bytes.size() - header_start) {
    throw std::runtime_error("truncated npy header");
  }

  const std::string header(
      reinterpret_cast<const char*>(bytes.data() + header_start), header_length);
  auto quoted_value = [&](const std::string& key) {
    const size_t key_pos = header.find(key);
    if (key_pos == std::string::npos) {
      throw std::runtime_error("missing npy header key: " + key);
    }
    const size_t colon = header.find(':', key_pos + key.size());
    const size_t first_quote = header.find_first_of("'\"", colon);
    if (colon == std::string::npos || first_quote == std::string::npos) {
      throw std::runtime_error("bad npy header key: " + key);
    }
    const size_t second_quote = header.find(header[first_quote], first_quote + 1);
    if (second_quote == std::string::npos) {
      throw std::runtime_error("unterminated npy header value: " + key);
    }
    return header.substr(first_quote + 1, second_quote - first_quote - 1);
  };

  const size_t fortran_pos = header.find("fortran_order");
  if (fortran_pos == std::string::npos) {
    throw std::runtime_error("missing npy fortran_order");
  }
  const size_t fortran_colon = header.find(':', fortran_pos);
  const size_t fortran_end = header.find(',', fortran_colon);
  if (fortran_colon == std::string::npos ||
      trim(header.substr(fortran_colon + 1, fortran_end - fortran_colon - 1)) != "False") {
    throw std::runtime_error("Fortran-order npy unsupported");
  }

  const size_t shape_pos = header.find("shape");
  const size_t left = header.find('(', shape_pos);
  const size_t right = header.find(')', left);
  if (shape_pos == std::string::npos || left == std::string::npos || right == std::string::npos) {
    throw std::runtime_error("bad npy shape");
  }

  ArrayView result;
  result.descr = quoted_value("descr");
  if (result.descr.empty() || result.descr[0] == '>') {
    throw std::runtime_error("big-endian npy dtype unsupported: " + result.descr);
  }
  std::stringstream shape_stream(header.substr(left + 1, right - left - 1));
  std::string token;
  while (std::getline(shape_stream, token, ',')) {
    token = trim(token);
    if (token.empty()) continue;
    if (token[0] == '-') throw std::runtime_error("negative npy shape dimension");
    size_t consumed = 0;
    const unsigned long long dimension = std::stoull(token, &consumed);
    if (consumed != token.size() || dimension > std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("invalid npy shape dimension: " + token);
    }
    result.shape.push_back(static_cast<size_t>(dimension));
  }
  result.data = bytes.data() + header_start + header_length;
  result.data_bytes = bytes.size() - header_start - header_length;
  return result;
}

template <typename T>
inline T read_unaligned(const uint8_t* data) {
  static_assert(std::is_trivially_copyable<T>::value, "unaligned read requires POD data");
  T value{};
  std::memcpy(&value, data, sizeof(T));
  return value;
}

inline void require_payload_size(const ArrayView& array, size_t element_size,
                                 const std::string& name) {
  const size_t count = element_count(array.shape);
  if (count != 0 && element_size > std::numeric_limits<size_t>::max() / count) {
    throw std::runtime_error("npy payload size overflows size_t: " + name);
  }
  const size_t expected = count * element_size;
  if (array.data_bytes != expected) {
    throw std::runtime_error("npy payload size mismatch for " + name + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(array.data_bytes));
  }
}

inline std::vector<float> to_float_vector(const std::vector<uint8_t>& bytes,
                                          const std::string& name,
                                          bool allow_integers = false) {
  const ArrayView array = parse_header(bytes);
  const size_t count = element_count(array.shape);
  std::vector<float> output(count);
  if (array.descr == "<f4" || array.descr == "|f4") {
    require_payload_size(array, sizeof(float), name);
    std::memcpy(output.data(), array.data, array.data_bytes);
  } else if (array.descr == "<f8" || array.descr == "|f8") {
    require_payload_size(array, sizeof(double), name);
    for (size_t index = 0; index < count; ++index) {
      output[index] = static_cast<float>(read_unaligned<double>(array.data + index * sizeof(double)));
    }
  } else if (allow_integers && array.descr == "<i8") {
    require_payload_size(array, sizeof(int64_t), name);
    for (size_t index = 0; index < count; ++index) {
      output[index] = static_cast<float>(read_unaligned<int64_t>(array.data + index * sizeof(int64_t)));
    }
  } else if (allow_integers && array.descr == "<i4") {
    require_payload_size(array, sizeof(int32_t), name);
    for (size_t index = 0; index < count; ++index) {
      output[index] = static_cast<float>(read_unaligned<int32_t>(array.data + index * sizeof(int32_t)));
    }
  } else {
    throw std::runtime_error("unsupported npy dtype for " + name + ": " + array.descr);
  }
  return output;
}

inline int64_t to_int64_scalar(const std::vector<uint8_t>& bytes,
                               const std::string& name) {
  const ArrayView array = parse_header(bytes);
  if (element_count(array.shape) != 1) {
    throw std::runtime_error("npy scalar expected: " + name);
  }
  if (array.descr == "<i8") {
    require_payload_size(array, sizeof(int64_t), name);
    return read_unaligned<int64_t>(array.data);
  }
  if (array.descr == "<i4") {
    require_payload_size(array, sizeof(int32_t), name);
    return read_unaligned<int32_t>(array.data);
  }
  throw std::runtime_error("unsupported integer npy dtype for " + name + ": " + array.descr);
}

inline float to_float_scalar(const std::vector<uint8_t>& bytes,
                             const std::string& name) {
  const auto values = to_float_vector(bytes, name, true);
  if (values.size() != 1) throw std::runtime_error("npy scalar expected: " + name);
  return values.front();
}

inline ArrayView metadata(
    const std::unordered_map<std::string, std::vector<uint8_t>>& archive,
    const std::string& key) {
  const auto found = archive.find(key);
  if (found == archive.end()) throw std::runtime_error("missing npz entry: " + key);
  return parse_header(found->second);
}

}  // namespace action_go::runtime::npy
