#include "action_go/runtime/onnx_runner.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace action_go::runtime {
namespace {

std::string shape_string(const std::vector<int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) oss << ",";
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

void require_finite_vector(const std::vector<float>& values, const std::string& name) {
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error(name + " contains NaN/Inf at index " + std::to_string(i));
    }
  }
}

size_t checked_shape_elements(const std::vector<int64_t>& shape, const std::string& name) {
  size_t count = 1;
  for (const int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error(name + " has dynamic/negative runtime tensor dim " + std::to_string(dim));
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::string lower_string(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool contains_provider(const std::vector<std::string>& providers, const std::string& provider) {
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

std::vector<float> checked_output_vector(const Ort::Value& value,
                                         const std::string& output_name,
                                         int expected_dim) {
  auto info = value.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  const size_t actual = info.GetElementCount();
  if (expected_dim < 0) {
    const float* p = value.GetTensorData<float>();
    std::vector<float> out(p, p + actual);
    require_finite_vector(out, "ONNX output '" + output_name + "'");
    return out;
  }
  const size_t expected = static_cast<size_t>(expected_dim);
  if (actual < expected) {
    throw std::runtime_error("ONNX output '" + output_name + "' too small: expected exactly " +
                             std::to_string(expected) + " floats, got " + std::to_string(actual) +
                             " shape=" + shape_string(shape));
  }
  if (actual > expected) {
    throw std::runtime_error("ONNX output '" + output_name + "' too large: expected exactly " +
                             std::to_string(expected) + " floats, got " + std::to_string(actual) +
                             " shape=" + shape_string(shape));
  }
  const float* p = value.GetTensorData<float>();
  std::vector<float> out(p, p + expected);
  require_finite_vector(out, "ONNX output '" + output_name + "'");
  return out;
}

}  // namespace

void OnnxRunner::load(const std::filesystem::path& model, int threads) {
  LoadOptions options;
  options.threads = threads;
  load(model, options);
}

void OnnxRunner::load(const std::filesystem::path& model, const LoadOptions& config) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  options.SetIntraOpNumThreads(std::max(1, config.threads));
  options.SetInterOpNumThreads(1);

  const auto providers = Ort::GetAvailableProviders();
  const bool has_cuda = contains_provider(providers, "CUDAExecutionProvider");
  const bool has_tensorrt = contains_provider(providers, "TensorrtExecutionProvider");
  const std::string requested = lower_string(config.execution_provider);
  selected_execution_provider_ = "cpu";

  auto append_cuda = [&]() {
    OrtCUDAProviderOptions cuda{};
    cuda.device_id = config.device_id;
    options.AppendExecutionProvider_CUDA(cuda);
    selected_execution_provider_ = "cuda";
  };
  auto append_tensorrt = [&]() {
    if (!config.tensorrt_cache_dir.empty()) {
      std::filesystem::create_directories(config.tensorrt_cache_dir);
    }
    OrtTensorRTProviderOptions tensorrt{};
    tensorrt.device_id = config.device_id;
    tensorrt.trt_max_partition_iterations = 1000;
    tensorrt.trt_min_subgraph_size = 1;
    tensorrt.trt_max_workspace_size = config.tensorrt_workspace_size;
    tensorrt.trt_fp16_enable = config.tensorrt_fp16 ? 1 : 0;
    tensorrt.trt_engine_cache_enable = config.tensorrt_cache_dir.empty() ? 0 : 1;
    const std::string cache = config.tensorrt_cache_dir.string();
    tensorrt.trt_engine_cache_path = cache.empty() ? nullptr : cache.c_str();
    tensorrt.trt_force_sequential_engine_build = 1;
    options.AppendExecutionProvider_TensorRT(tensorrt);
    selected_execution_provider_ = "tensorrt";
  };

  if (requested == "cuda") {
    if (!has_cuda) throw std::runtime_error("CUDAExecutionProvider is unavailable");
    append_cuda();
  } else if (requested == "tensorrt" || requested == "trt") {
    if (!has_tensorrt) throw std::runtime_error("TensorrtExecutionProvider is unavailable");
    append_tensorrt();
    if (has_cuda) append_cuda();
    selected_execution_provider_ = "tensorrt";
  } else if (requested == "auto") {
    if (has_tensorrt) {
      append_tensorrt();
      if (has_cuda) append_cuda();
      selected_execution_provider_ = "tensorrt";
    } else if (has_cuda) {
      append_cuda();
    }
  } else if (requested != "cpu") {
    throw std::runtime_error("unknown ONNX execution provider: " + config.execution_provider);
  }

  session_ = Ort::Session(env_, model.string().c_str(), options);
  Ort::AllocatorWithDefaultOptions alloc;
  input_names_.clear();
  output_names_.clear();
  output_idx_.clear();
  input_shapes_.clear();
  for (size_t i = 0; i < session_.GetInputCount(); ++i) {
    auto n = session_.GetInputNameAllocated(i, alloc);
    input_names_.emplace_back(n.get());
    input_shapes_.push_back(session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
    auto n = session_.GetOutputNameAllocated(i, alloc);
    output_idx_[n.get()] = output_names_.size();
    output_names_.emplace_back(n.get());
  }
  if (input_names_.empty() || output_names_.empty()) throw std::runtime_error("invalid ONNX IO");
}

void OnnxRunner::require_inputs(const std::vector<std::string>& names) const {
  for (const std::string& name : names) {
    if (name.empty()) continue;
    if (std::find(input_names_.begin(), input_names_.end(), name) == input_names_.end()) {
      throw std::runtime_error("ONNX missing input: " + name);
    }
  }
}

void OnnxRunner::require_outputs(const std::vector<std::string>& names) const {
  for (const std::string& name : names) {
    if (name.empty()) continue;
    if (output_idx_.find(name) == output_idx_.end()) {
      throw std::runtime_error("ONNX missing output: " + name);
    }
  }
}

int OnnxRunner::inferred_obs_dim(int fallback) const {
  if (!input_shapes_.empty() && input_shapes_[0].size() >= 2 && input_shapes_[0][1] > 0) {
    return static_cast<int>(input_shapes_[0][1]);
  }
  return fallback;
}

std::vector<float> OnnxRunner::run_single(const std::vector<float>& obs,
                                          const std::string& input_name,
                                          const std::string& output_name,
                                          int output_dim) {
  require_finite_vector(obs, "ONNX input observation");
  const std::string in_name = input_name.empty() ? input_names_.at(0) : input_name;
  const std::string out_name = output_name.empty() ? output_names_.at(0) : output_name;
  std::array<int64_t, 2> shape{1, static_cast<int64_t>(obs.size())};
  Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
      mem_, const_cast<float*>(obs.data()), obs.size(), shape.data(), shape.size());
  const char* in = in_name.c_str();
  const char* out = out_name.c_str();
  auto outs = session_.Run(Ort::RunOptions{nullptr}, &in, &obs_tensor, 1, &out, 1);
  return checked_output_vector(outs[0], out_name, output_dim);
}

std::unordered_map<std::string, std::vector<float>> OnnxRunner::run_obs_time(
    const std::vector<float>& obs,
    float time_step,
    const std::string& obs_input,
    const std::string& time_input,
    const std::unordered_map<std::string, int>& output_dims) {
  require_finite_vector(obs, "ONNX input observation");
  if (!std::isfinite(time_step)) throw std::runtime_error("ONNX time_step input contains NaN/Inf");

  std::array<int64_t, 2> obs_shape{1, static_cast<int64_t>(obs.size())};
  std::array<int64_t, 2> t_shape{1, 1};
  std::array<float, 1> tbuf{time_step};
  Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
      mem_, const_cast<float*>(obs.data()), obs.size(), obs_shape.data(), obs_shape.size());
  Ort::Value t_tensor = Ort::Value::CreateTensor<float>(
      mem_, tbuf.data(), tbuf.size(), t_shape.data(), t_shape.size());

  std::vector<Ort::Value> inputs;
  std::vector<const char*> in_names;
  inputs.reserve(input_names_.size());
  in_names.reserve(input_names_.size());
  bool used_obs = false;
  bool used_time = false;
  for (const std::string& name : input_names_) {
    in_names.push_back(name.c_str());
    if (name == obs_input) {
      inputs.emplace_back(std::move(obs_tensor));
      used_obs = true;
    } else if (name == time_input) {
      inputs.emplace_back(std::move(t_tensor));
      used_time = true;
    } else {
      throw std::runtime_error("unsupported ONNX input for embedded_mimic_time: " + name);
    }
  }
  if (!used_obs || !used_time) throw std::runtime_error("obs/time input not bound");

  std::vector<std::string> requested_names;
  requested_names.reserve(output_dims.size());
  for (const auto& kv : output_dims) requested_names.push_back(kv.first);
  std::sort(requested_names.begin(), requested_names.end());
  std::vector<const char*> out_names;
  out_names.reserve(requested_names.size());
  for (const std::string& name : requested_names) out_names.push_back(name.c_str());

  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
                           out_names.data(), out_names.size());
  std::unordered_map<std::string, std::vector<float>> result;
  for (size_t i = 0; i < requested_names.size(); ++i) {
    const std::string& name = requested_names[i];
    result[name] = checked_output_vector(outs[i], name, output_dims.at(name));
  }
  return result;
}

std::unordered_map<std::string, std::vector<float>> OnnxRunner::run_named(
    const std::unordered_map<std::string, TensorInput>& named_inputs,
    const std::unordered_map<std::string, int>& output_dims) {
  std::vector<Ort::Value> inputs;
  std::vector<const char*> in_names;
  inputs.reserve(input_names_.size());
  in_names.reserve(input_names_.size());
  for (const std::string& name : input_names_) {
    const auto it = named_inputs.find(name);
    if (it == named_inputs.end()) throw std::runtime_error("missing named ONNX input: " + name);
    const auto& tensor = it->second;
    require_finite_vector(tensor.values, "ONNX named input '" + name + "'");
    const size_t expected_values = checked_shape_elements(tensor.shape, "ONNX named input '" + name + "'");
    if (expected_values != tensor.values.size()) {
      throw std::runtime_error("ONNX named input '" + name + "' shape=" + shape_string(tensor.shape) +
                               " contains " + std::to_string(expected_values) +
                               " elements but values.size=" + std::to_string(tensor.values.size()));
    }
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, const_cast<float*>(tensor.values.data()), tensor.values.size(),
        const_cast<int64_t*>(tensor.shape.data()), tensor.shape.size()));
    in_names.push_back(name.c_str());
  }

  std::vector<std::string> requested_names;
  requested_names.reserve(output_dims.size());
  for (const auto& kv : output_dims) requested_names.push_back(kv.first);
  std::sort(requested_names.begin(), requested_names.end());
  std::vector<const char*> out_names;
  out_names.reserve(requested_names.size());
  for (const std::string& name : requested_names) out_names.push_back(name.c_str());

  auto outs = session_.Run(Ort::RunOptions{nullptr}, in_names.data(), inputs.data(), inputs.size(),
                           out_names.data(), out_names.size());
  std::unordered_map<std::string, std::vector<float>> result;
  for (size_t i = 0; i < requested_names.size(); ++i) {
    const std::string& name = requested_names[i];
    result[name] = checked_output_vector(outs[i], name, output_dims.at(name));
  }
  return result;
}

}  // namespace action_go::runtime
