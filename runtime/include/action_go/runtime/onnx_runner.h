#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace action_go::runtime {

class OnnxRunner {
 public:
  struct LoadOptions {
    int threads{1};
    std::string execution_provider{"cpu"};
    int device_id{0};
    std::filesystem::path tensorrt_cache_dir;
    size_t tensorrt_workspace_size{1ULL << 30};
    bool tensorrt_fp16{false};
  };

  struct TensorInput {
    std::vector<float> values;
    std::vector<int64_t> shape;
  };

  void load(const std::filesystem::path& model, int threads);
  void load(const std::filesystem::path& model, const LoadOptions& options);

  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }
  const std::vector<std::vector<int64_t>>& input_shapes() const { return input_shapes_; }
  const std::string& selected_execution_provider() const { return selected_execution_provider_; }

  void require_inputs(const std::vector<std::string>& names) const;
  void require_outputs(const std::vector<std::string>& names) const;

  int inferred_obs_dim(int fallback) const;

  std::vector<float> run_single(const std::vector<float>& obs,
                                const std::string& input_name,
                                const std::string& output_name,
                                int output_dim);

  std::unordered_map<std::string, std::vector<float>> run_obs_time(
      const std::vector<float>& obs,
      float time_step,
      const std::string& obs_input,
      const std::string& time_input,
      const std::unordered_map<std::string, int>& output_dims);

  std::unordered_map<std::string, std::vector<float>> run_named(
      const std::unordered_map<std::string, TensorInput>& named_inputs,
      const std::unordered_map<std::string, int>& output_dims);

 private:
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "generic_policy_player"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, size_t> output_idx_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::string selected_execution_provider_{"cpu"};
};

}  // namespace action_go::runtime
