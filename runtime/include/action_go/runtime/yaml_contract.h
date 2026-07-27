#pragma once

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace action_go::runtime::yaml_contract {

inline std::vector<std::string> split_path(const std::string& path) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : path) {
    if (c == '.') {
      if (!cur.empty()) parts.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) parts.push_back(cur);
  return parts;
}

inline YAML::Node optional_node(const YAML::Node& root, const std::string& path) {
  YAML::Node cur = root;
  for (const std::string& key : split_path(path)) {
    if (!cur || !cur.IsMap()) return YAML::Node();
    YAML::Node next;
    for (YAML::const_iterator it = cur.begin(); it != cur.end(); ++it) {
      if (!it->first.IsScalar()) continue;
      if (it->first.as<std::string>() == key) {
        next = it->second;
        break;
      }
    }
    if (!next || !next.IsDefined() || next.IsNull()) return YAML::Node();
    cur.reset(next);
  }
  return cur;
}

inline YAML::Node required_node(const YAML::Node& root, const std::string& path) {
  YAML::Node node = optional_node(root, path);
  if (!node || !node.IsDefined() || node.IsNull()) {
    throw std::runtime_error("missing required YAML field: " + path);
  }
  return node;
}

template <typename T>
T required_scalar(const YAML::Node& root, const std::string& path) {
  const YAML::Node node = required_node(root, path);
  if (!node.IsScalar()) {
    throw std::runtime_error("YAML field must be scalar: " + path);
  }
  return node.as<T>();
}

template <typename T>
T optional_scalar(const YAML::Node& root, const std::string& path, const T& fallback) {
  const YAML::Node node = optional_node(root, path);
  if (!node) return fallback;
  if (!node.IsScalar()) {
    throw std::runtime_error("YAML field must be scalar: " + path);
  }
  return node.as<T>();
}

template <typename T>
std::vector<T> required_vector(const YAML::Node& root, const std::string& path) {
  const YAML::Node node = required_node(root, path);
  if (!node.IsSequence()) {
    throw std::runtime_error("YAML field must be a sequence: " + path);
  }
  std::vector<T> out;
  out.reserve(node.size());
  for (const YAML::Node& value : node) out.push_back(value.as<T>());
  return out;
}

template <typename T>
std::vector<T> optional_vector(const YAML::Node& root, const std::string& path) {
  const YAML::Node node = optional_node(root, path);
  if (!node) return {};
  if (!node.IsSequence()) {
    throw std::runtime_error("YAML field must be a sequence: " + path);
  }
  std::vector<T> out;
  out.reserve(node.size());
  for (const YAML::Node& value : node) out.push_back(value.as<T>());
  return out;
}

inline void require_size(const std::string& path, std::size_t expected, std::size_t actual) {
  if (expected != actual) {
    throw std::runtime_error(path + " size mismatch: expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

template <typename T>
std::vector<T> required_vector_size(const YAML::Node& root,
                                    const std::string& path,
                                    std::size_t expected) {
  auto values = required_vector<T>(root, path);
  require_size(path, expected, values.size());
  return values;
}

inline void require_permutation(const std::string& path, const std::vector<int>& values, int n) {
  require_size(path, static_cast<std::size_t>(n), values.size());
  std::vector<int> seen(static_cast<std::size_t>(n), 0);
  for (int value : values) {
    if (value < 0 || value >= n) {
      throw std::runtime_error(path + " contains out-of-range index " + std::to_string(value));
    }
    if (++seen[static_cast<std::size_t>(value)] > 1) {
      throw std::runtime_error(path + " is not a permutation; duplicate index " + std::to_string(value));
    }
  }
}

}  // namespace action_go::runtime::yaml_contract
