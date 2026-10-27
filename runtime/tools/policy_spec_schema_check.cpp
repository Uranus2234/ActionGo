#include "action_go/runtime/onnx_runner.h"
#include "action_go/runtime/policy_contract.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/pbfm_policy.h"
#include "action_go/runtime/yaml_contract.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace yamlc = action_go::runtime::yaml_contract;

using action_go::runtime::GenericPlayerKind;
using action_go::runtime::OnnxRunner;
using action_go::runtime::PolicyContract;
using action_go::runtime::PolicyProtocolConstraint;
using action_go::runtime::PolicyTemplateId;
using action_go::runtime::PolicyYamlDimRule;
using action_go::runtime::kSonicDecoderObsDim;
using action_go::runtime::kSonicEncoderObsDim;
using action_go::runtime::kSonicV11EncoderObsDim;
using action_go::runtime::kSonicHistoryLen;
using action_go::runtime::kSonicPlannerContextFrames;
using action_go::runtime::kSonicPlannerModeCount;
using action_go::runtime::kSonicPlannerOutputFrames;
using action_go::runtime::kSonicPlannerQposDim;
using action_go::runtime::kSonicReferenceFrames;
using action_go::runtime::kSonicReferenceStep;
using action_go::runtime::kSonicSmplJointCount;
using action_go::runtime::kSonicSmplPoseCount;
using action_go::runtime::kSonicSmplReferenceFrames;
using action_go::runtime::kSonicTokenDim;
using action_go::runtime::kSonicZmqHeaderSize;
using action_go::runtime::contract_for_template;
using action_go::runtime::format_supported_templates;
using action_go::runtime::format_supported_variants_for_template;
using action_go::runtime::format_protocol_constraint_summary;
using action_go::runtime::player_binary_name;
using action_go::runtime::protocol_constraint_for_variant;
using action_go::runtime::protocol_yaml_dim_rules_for_variant;
using action_go::runtime::protocol_names;
using action_go::runtime::variant_contract;
using action_go::runtime::variant_allowed_for_template;

namespace {

struct Args {
  bool all{false};
  bool strict_onnx{true};
  bool allow_missing_assets{false};
  bool installed_only{false};
  std::vector<fs::path> specs;
};

bool g_allow_missing_assets = false;
size_t g_missing_assets = 0;

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool present(const YAML::Node& node) {
  return node && node.IsDefined() && !node.IsNull();
}

int sonic_encoder_obs_dim_for_spec(const YAML::Node& y) {
  const YAML::Node checkpoint_node = yamlc::optional_node(y, "runtime.checkpoint");
  const std::string checkpoint = present(checkpoint_node)
      ? lower(checkpoint_node.as<std::string>())
      : "default";
  if (checkpoint == "default") return kSonicEncoderObsDim;
  if (checkpoint == "sonic_v1_1") return kSonicV11EncoderObsDim;
  throw std::runtime_error("unsupported SONIC runtime.checkpoint: " + checkpoint);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--all") {
      args.all = true;
    } else if (key == "--spec") {
      if (i + 1 >= argc) throw std::runtime_error("missing value for --spec");
      args.specs.emplace_back(argv[++i]);
    } else if (key == "--no-onnx") {
      args.strict_onnx = false;
    } else if (key == "--allow-missing-assets") {
      args.allow_missing_assets = true;
      args.strict_onnx = false;
    } else if (key == "--installed-only") {
      args.installed_only = true;
      args.allow_missing_assets = true;
    } else if (key == "--help") {
      std::cout << "Usage:\n"
                << "  policy_spec_schema_check --all [--no-onnx] [--allow-missing-assets|--installed-only]\n"
                << "  policy_spec_schema_check --spec actions/<policy>/config/generic_<policy>.yaml\n\n"
                << "  --allow-missing-assets  validate adapter/config semantics without opening ONNX files\n"
                << "  --installed-only        allow absent assets and validate real I/O for installed ONNX files\n";
      std::exit(0);
    } else {
      args.specs.emplace_back(key);
    }
  }
  if (!args.all && args.specs.empty()) throw std::runtime_error("use --all or --spec <path>");
  return args;
}

fs::path resolve_relative(const fs::path& base_file, const std::string& raw) {
  if (raw.empty()) return {};
  fs::path p(raw);
  if (p.is_absolute()) return p.lexically_normal();
  if (fs::exists(p)) return fs::absolute(p).lexically_normal();
  return (base_file.parent_path() / p).lexically_normal();
}

bool looks_like_git_lfs_pointer(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) return false;
  char buf[256]{};
  in.read(buf, sizeof(buf) - 1);
  const std::string head(buf, static_cast<size_t>(std::max<std::streamsize>(0, in.gcount())));
  return head.find("version https://git-lfs.github.com/spec/v1") != std::string::npos;
}

fs::path require_existing_path(const fs::path& spec, const std::string& raw,
                               const std::string& field, bool optional_asset = false) {
  const fs::path path = resolve_relative(spec, raw);
  if (path.empty()) throw std::runtime_error(field + " is empty");
  if (!fs::exists(path)) {
    if (optional_asset && g_allow_missing_assets) {
      ++g_missing_assets;
      std::cout << "    asset[missing] " << field << "=" << path.string() << "\n";
      return path;
    }
    throw std::runtime_error(field + " not found: " + path.string());
  }
  if (looks_like_git_lfs_pointer(path)) {
    if (optional_asset && g_allow_missing_assets) {
      ++g_missing_assets;
      std::cout << "    asset[missing-lfs] " << field << "=" << path.string() << "\n";
      return path;
    }
    throw std::runtime_error(field + " is a Git LFS pointer, not a real asset: " + path.string());
  }
  return path;
}

int require_positive_int(const YAML::Node& root, const std::string& path) {
  const int value = yamlc::required_scalar<int>(root, path);
  if (value <= 0) throw std::runtime_error(path + " must be positive");
  return value;
}

std::optional<int> optional_int(const YAML::Node& root, const std::string& path) {
  const YAML::Node node = yamlc::optional_node(root, path);
  if (!present(node)) return std::nullopt;
  if (!node.IsScalar()) throw std::runtime_error(path + " must be scalar");
  const std::string raw = lower(node.as<std::string>());
  if (raw.empty() || raw == "auto") return std::nullopt;
  return node.as<int>();
}

std::vector<std::string> require_string_vector(const YAML::Node& root, const std::string& path) {
  auto values = yamlc::required_vector<std::string>(root, path);
  if (values.empty()) throw std::runtime_error(path + " must not be empty");
  for (const std::string& value : values) {
    if (value.empty()) throw std::runtime_error(path + " contains an empty name");
  }
  return values;
}

std::string join_names(const std::vector<std::string>& values) {
  std::ostringstream out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ", ";
    out << values[i];
  }
  return out.str();
}

void require_name_list_equal(const std::string& field,
                             const std::vector<std::string>& actual,
                             const std::vector<const char*>& expected_raw) {
  if (expected_raw.empty()) return;
  const std::vector<std::string> expected = protocol_names(expected_raw);
  if (actual != expected) {
    throw std::runtime_error(field + " mismatch: expected [" + join_names(expected) +
                             "], got [" + join_names(actual) + "]");
  }
}

template <typename T>
std::vector<T> require_vector_one_of(const YAML::Node& root,
                                     const std::vector<std::string>& paths,
                                     size_t expected,
                                     const std::string& label) {
  for (const std::string& path : paths) {
    const YAML::Node node = yamlc::optional_node(root, path);
    if (!present(node)) continue;
    auto values = yamlc::required_vector<T>(root, path);
    yamlc::require_size(path, expected, values.size());
    return values;
  }
  std::ostringstream oss;
  oss << "missing required vector for " << label << " at one of: ";
  for (size_t i = 0; i < paths.size(); ++i) {
    if (i) oss << ", ";
    oss << paths[i];
  }
  throw std::runtime_error(oss.str());
}

int require_int_one_of(const YAML::Node& root,
                       const std::vector<std::string>& paths,
                       const std::string& label) {
  for (const std::string& path : paths) {
    const YAML::Node node = yamlc::optional_node(root, path);
    if (present(node)) return yamlc::required_scalar<int>(root, path);
  }
  std::ostringstream oss;
  oss << "missing required scalar for " << label << " at one of: ";
  for (size_t i = 0; i < paths.size(); ++i) {
    if (i) oss << ", ";
    oss << paths[i];
  }
  throw std::runtime_error(oss.str());
}

void require_matching_int(const YAML::Node& root,
                          const std::vector<std::string>& paths,
                          int expected,
                          const std::string& label) {
  const int actual = require_int_one_of(root, paths, label);
  if (actual != expected) {
    throw std::runtime_error(label + " mismatch: expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

void validate_protocol_yaml_dim_rules(const YAML::Node& root,
                                      const std::string& variant) {
  for (const PolicyYamlDimRule& rule : protocol_yaml_dim_rules_for_variant(variant)) {
    require_matching_int(root, {rule.path}, rule.expected, rule.label);
  }
}

void require_permutation_one_of(const YAML::Node& root,
                                const std::vector<std::string>& paths,
                                int n,
                                const std::string& label) {
  auto values = require_vector_one_of<int>(root, paths, static_cast<size_t>(n), label);
  yamlc::require_permutation(label, values, n);
}

void validate_standard_robot_vectors(const YAML::Node& params,
                                     int motor_count,
                                     bool require_default_q,
                                     bool require_action_scale,
                                     const std::string& joint_order_path = "robot.joint_order") {
  require_vector_one_of<std::string>(params, {joint_order_path}, static_cast<size_t>(motor_count), joint_order_path);
  if (require_default_q) {
    require_vector_one_of<float>(params, {"robot.default_angles", "default_angles"},
                                 static_cast<size_t>(motor_count), "default angles");
  }
  if (require_action_scale) {
    require_vector_one_of<float>(params, {"robot.action_scale", "action_scale"},
                                 static_cast<size_t>(motor_count), "action scale");
  }
  require_vector_one_of<float>(params, {"robot.kp", "kp", "kps"}, static_cast<size_t>(motor_count), "kp");
  require_vector_one_of<float>(params, {"robot.kd", "kd", "kds"}, static_cast<size_t>(motor_count), "kd");
  require_vector_one_of<float>(params, {"robot.torque_limits", "torque_limits", "tau_limit"},
                               static_cast<size_t>(motor_count), "torque limits");
  require_vector_one_of<float>(params, {"robot.joint_pos_lower_limit", "joint_pos_lower_limit", "joint_limits_lower"},
                               static_cast<size_t>(motor_count), "joint lower limits");
  require_vector_one_of<float>(params, {"robot.joint_pos_upper_limit", "joint_pos_upper_limit", "joint_limits_upper"},
                               static_cast<size_t>(motor_count), "joint upper limits");
}

void validate_onnx_io(const fs::path& model_path,
                      const std::vector<std::string>& inputs,
                      const std::vector<std::string>& outputs,
                      bool strict_onnx,
                      const std::string& label) {
  if (!strict_onnx) return;
  if (!fs::exists(model_path)) return;
  OnnxRunner runner;
  runner.load(model_path, 1);
  runner.require_inputs(inputs);
  runner.require_outputs(outputs);
  std::cout << "    onnx[" << label << "] inputs=" << inputs.size()
            << " outputs=" << outputs.size() << "\n";
}

void validate_model(const fs::path& spec,
                    const YAML::Node& y,
                    bool strict_onnx,
                    const PolicyContract& contract,
                    const PolicyProtocolConstraint& protocol) {
  const YAML::Node model = yamlc::required_node(y, "model");
  bool checked_any = false;

  if (present(yamlc::optional_node(model, "path"))) {
    const fs::path path = require_existing_path(
        spec, yamlc::required_scalar<std::string>(model, "path"), "model.path", true);
    const auto inputs = require_string_vector(model, "input_names");
    const auto outputs = require_string_vector(model, "output_names");
    require_name_list_equal("model.input_names", inputs, protocol.model_input_names);
    require_name_list_equal("model.output_names", outputs, protocol.model_output_names);
    validate_onnx_io(path, inputs, outputs, strict_onnx, "model.path");
    checked_any = true;
  }

  const YAML::Node sessions = yamlc::optional_node(model, "sessions");
  if (!protocol.sessions.empty() && !present(sessions)) {
    throw std::runtime_error("model.sessions is required by protocol constraint");
  }
  if (present(sessions)) {
    if (!sessions.IsMap()) throw std::runtime_error("model.sessions must be a map");
    for (const auto& expected : protocol.sessions) {
      if (!present(sessions[expected.name])) {
        throw std::runtime_error("model.sessions missing protocol session: " + std::string(expected.name));
      }
    }
    for (YAML::const_iterator it = sessions.begin(); it != sessions.end(); ++it) {
      const std::string name = it->first.as<std::string>();
      const YAML::Node session = it->second;
      const fs::path path = require_existing_path(spec, yamlc::required_scalar<std::string>(session, "path"),
                                                  "model.sessions." + name + ".path", true);
      const auto inputs = require_string_vector(session, "input_names");
      const auto outputs = require_string_vector(session, "output_names");
      for (const auto& expected : protocol.sessions) {
        if (name == expected.name) {
          require_name_list_equal("model.sessions." + name + ".input_names", inputs, expected.input_names);
          require_name_list_equal("model.sessions." + name + ".output_names", outputs, expected.output_names);
        }
      }
      validate_onnx_io(path, inputs, outputs, strict_onnx, name);
      checked_any = true;
    }
  }

  if (contract.id == PolicyTemplateId::kDepthConditionedActor &&
      yamlc::required_scalar<std::string>(y, "runtime.variant") != "pbfm") {
    require_existing_path(spec, yamlc::required_scalar<std::string>(model, "logdir"), "model.logdir", true);
    if (present(yamlc::optional_node(model, "standdir"))) {
      require_existing_path(spec, yamlc::required_scalar<std::string>(model, "standdir"), "model.standdir", true);
    }
  } else if (contract.id == PolicyTemplateId::kCompositeMotionPolicy) {
    require_existing_path(spec, yamlc::required_scalar<std::string>(model, "base"), "model.base", true);
    require_existing_path(spec, yamlc::required_scalar<std::string>(model, "residual"), "model.residual", true);
    require_existing_path(spec, yamlc::required_scalar<std::string>(model, "fk"), "model.fk", true);
  }

  if (!checked_any) {
    throw std::runtime_error("model must define model.path with input/output names or model.sessions");
  }
}

void validate_source_sections(const fs::path& spec,
                              const YAML::Node& y,
                              const PolicyContract& contract,
                              const PolicyProtocolConstraint& protocol) {
  switch (contract.id) {
    case PolicyTemplateId::kEmbeddedMimicTime:
      yamlc::required_scalar<std::string>(y, "mimic_time.time_input_name");
      yamlc::required_scalar<std::string>(y, "mimic_time.time_step_source");
      yamlc::required_vector<std::string>(y, "mimic_time.internal_reference_outputs");
      if (lower(yamlc::required_scalar<std::string>(y, "runtime.variant")) == "beyondmimic") {
        const int ref_body_count = yamlc::required_scalar<int>(y, "mimic_time.ref_body_count");
        const int anchor_body_index = yamlc::required_scalar<int>(y, "mimic_time.anchor_body_index");
        if (ref_body_count != protocol.ref_body_count ||
            anchor_body_index != protocol.ref_anchor_index) {
          throw std::runtime_error(
              "BeyondMimic mimic_time ref_body_count/anchor_body_index do not match protocol constraint");
        }
      }
      break;
    case PolicyTemplateId::kMotionConditionedTracking: {
      const std::string variant = lower(yamlc::required_scalar<std::string>(y, "runtime.variant"));
      if (variant == "supermimic" || variant == "uni_supermimic" || variant == "allmimic") {
        require_existing_path(spec, yamlc::required_scalar<std::string>(y, "reference.path"), "reference.path", true);
        if (yamlc::required_scalar<std::string>(y, "reference.type") !=
            (variant == "allmimic" ? "allmimic_csv_v1" : "supermimic_csv_v1")) {
          throw std::runtime_error(variant + " reference.type does not match its cache format");
        }
        require_matching_int(y, {"runtime.inference_hz"}, 50, "SuperMimic inference Hz");
      } else if (variant == "protomotions") {
        require_existing_path(spec, yamlc::required_scalar<std::string>(y, "reference.path"), "reference.path", true);
        yamlc::required_scalar<std::string>(y, "reference.anchor_source");
      } else if (variant == "twist") {
        yamlc::required_scalar<std::string>(y, "motion_command.source");
        const std::string source = lower(yamlc::required_scalar<std::string>(y, "motion_command.source"));
        if (source == "offline-pkl") {
          require_existing_path(spec, yamlc::required_scalar<std::string>(y, "motion_command.pkl"), "motion_command.pkl", true);
        } else if (source == "offline-cache") {
          require_existing_path(spec, yamlc::required_scalar<std::string>(y, "motion_command.cache"), "motion_command.cache", true);
        }
        require_matching_int(y, {"motion_command.mimic_dim"}, protocol.twist_mimic_dim, "TWIST motion_command.mimic_dim");
        require_matching_int(y, {"motion_command.wrist_dim"}, protocol.twist_wrist_dim, "TWIST motion_command.wrist_dim");
      } else if (variant == "humanoid_gpt") {
        yamlc::required_scalar<std::string>(y, "reference.source");
        if (lower(yamlc::required_scalar<std::string>(y, "reference.source")) == "offline-npz") {
          require_existing_path(spec, yamlc::required_scalar<std::string>(y, "reference.path"), "reference.path", true);
        }
        require_matching_int(y, {"reference.qpos_dim"}, protocol.humanoid_gpt_qpos_dim, "Humanoid-GPT reference.qpos_dim");
        require_matching_int(y, {"reference.qvel_dim"}, protocol.humanoid_gpt_qvel_dim, "Humanoid-GPT reference.qvel_dim");
        require_matching_int(y, {"reference.keypoint_count"}, protocol.humanoid_gpt_keypoint_count, "Humanoid-GPT reference.keypoint_count");
        require_matching_int(y, {"reference.hmcp_header_size"}, protocol.humanoid_gpt_hmcp_header_size, "Humanoid-GPT reference.hmcp_header_size");
      } else if (variant == "sonic") {
        const std::string source = lower(yamlc::required_scalar<std::string>(y, "reference.source"));
        if (source == "offline-csv") {
          require_existing_path(spec, yamlc::required_scalar<std::string>(y, "reference.path"), "reference.path", true);
        }
        require_existing_path(spec, yamlc::required_scalar<std::string>(y, "observation.config"),
                              "observation.config", false);
        require_matching_int(y, {"observation.encoder_obs_dim"},
                             sonic_encoder_obs_dim_for_spec(y), "SONIC encoder obs dim");
        require_matching_int(y, {"observation.token_dim"}, kSonicTokenDim, "SONIC token dim");
        require_matching_int(y, {"observation.history_len"}, kSonicHistoryLen, "SONIC history len");
        require_matching_int(y, {"observation.smpl_joint_count"}, kSonicSmplJointCount, "SONIC SMPL joint count");
        require_matching_int(y, {"observation.smpl_pose_count"}, kSonicSmplPoseCount, "SONIC SMPL pose count");
        require_matching_int(y, {"observation.smpl_reference_frames"}, kSonicSmplReferenceFrames, "SONIC SMPL reference frames");
        require_matching_int(y, {"reference.future_frames"}, kSonicReferenceFrames, "SONIC future frames");
        require_matching_int(y, {"reference.future_step"}, kSonicReferenceStep, "SONIC future step");
        require_matching_int(y, {"reference.zmq_header_size"}, kSonicZmqHeaderSize, "SONIC ZMQ header size");
        const auto protocol_versions = yamlc::required_vector<int>(y, "reference.zmq_protocol_versions");
        if (protocol_versions != std::vector<int>({1, 2, 3})) {
          throw std::runtime_error("reference.zmq_protocol_versions must be [1, 2, 3]");
        }
        if (!yamlc::required_scalar<bool>(y, "planner.enabled")) {
          throw std::runtime_error("SONIC planner.enabled must be true for the public sonic variant");
        }
        require_matching_int(y, {"planner.version"}, 2, "SONIC planner version");
        require_matching_int(y, {"planner.inference_hz"}, 10, "SONIC planner inference Hz");
        require_matching_int(y, {"planner.output_hz"}, 30, "SONIC planner output Hz");
        require_matching_int(y, {"planner.reference_hz"}, 50, "SONIC planner reference Hz");
        require_matching_int(y, {"planner.context_frames"}, kSonicPlannerContextFrames, "SONIC planner context frames");
        require_matching_int(y, {"planner.context_qpos_dim"}, kSonicPlannerQposDim, "SONIC planner qpos dim");
        require_matching_int(y, {"planner.max_output_frames"}, kSonicPlannerOutputFrames, "SONIC planner output frames");
        require_matching_int(y, {"planner.min_output_frames"}, 24, "SONIC planner minimum output frames");
        require_matching_int(y, {"planner.mode_count"}, kSonicPlannerModeCount, "SONIC planner mode count");
        require_matching_int(y, {"planner.lookahead_steps"}, 2, "SONIC planner lookahead");
        require_matching_int(y, {"planner.blend_frames"}, 8, "SONIC planner blend frames");
        const float planner_height = yamlc::required_scalar<float>(y, "planner.default_height");
        if (!std::isfinite(planner_height) || std::abs(planner_height - 0.788740f) > 1e-6f) {
          throw std::runtime_error("planner.default_height differs from the V2 deployment value 0.788740");
        }
        if (yamlc::required_scalar<int64_t>(y, "planner.random_seed") != 1234) {
          throw std::runtime_error("planner.random_seed differs from the V2 deployment value 1234");
        }
        if (yamlc::required_vector<int64_t>(y, "planner.allowed_pred_num_tokens") !=
            std::vector<int64_t>({0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0})) {
          throw std::runtime_error("planner.allowed_pred_num_tokens differs from the V2 TensorRT deployment mask");
        }
      }
      break;
    }
    case PolicyTemplateId::kDepthConditionedActor:
      yamlc::required_scalar<std::string>(y, "sensor.depth_source");
      break;
    case PolicyTemplateId::kObjectConditionedActor:
      yamlc::required_scalar<std::string>(y, "object.source");
      break;
    case PolicyTemplateId::kCompositeMotionPolicy:
      require_existing_path(spec, yamlc::required_scalar<std::string>(y, "motion.path"), "motion.path", true);
      break;
    case PolicyTemplateId::kSingleActorState:
      break;
  }
}

void validate_params(const YAML::Node& y,
                     const YAML::Node& params,
                     const std::string& variant,
                     int motor_count,
                     int obs_dim,
                     int action_dim,
                     const PolicyProtocolConstraint& protocol) {
  if (present(yamlc::optional_node(params, "robot.motor_count"))) {
    require_matching_int(params, {"robot.motor_count"}, motor_count, "params robot.motor_count");
  }

  if (variant == "bfm_zero" || variant == "pbfm") {
    validate_standard_robot_vectors(params, motor_count, true, true);
    require_matching_int(params, {"policy.actor_obs_dim", "actor_obs_dim"}, obs_dim, "BFM-Zero actor obs dim");
    require_matching_int(y, {"observation.latent_dim"}, protocol.latent_dim, "BFM-Zero latent dim");
    require_matching_int(y, {"observation.history_steps"}, protocol.history_steps, "BFM-Zero history steps");
  } else if (variant == "unitree_rl_mjlab") {
    validate_standard_robot_vectors(params, motor_count, true, true);
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "unitree action dim");
    require_matching_int(y, {"observation.base_obs_dim"}, protocol.base_obs_dim, "unitree base obs dim");
    const auto allowed = yamlc::required_vector<int>(y, "observation.allowed_obs_dims");
    if (allowed != protocol.allowed_obs_dims) {
      throw std::runtime_error("unitree allowed obs dims mismatch");
    }
    yamlc::required_scalar<double>(params, "observation.phase_period");
  } else if (variant == "beyondmimic" || variant == "supermimic" || variant == "uni_supermimic" || variant == "allmimic") {
    validate_standard_robot_vectors(params, motor_count, true, true, "robot.policy_joint_order");
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "BeyondMimic action dim");
    require_matching_int(params, {"policy.num_obs", "num_obs"}, obs_dim, "BeyondMimic obs dim");
    require_permutation_one_of(params, {"robot.joint_map", "joint_index"}, motor_count, "BeyondMimic joint map");
    if (variant == "allmimic") {
      require_vector_one_of<float>(params, {"action.residual_scale"}, motor_count, "AllMimic residual scale");
      require_vector_one_of<float>(params, {"action.residual_reference"}, motor_count, "AllMimic residual reference");
    }
  } else if (variant == "holosoma") {
    validate_standard_robot_vectors(params, motor_count, true, true, "robot.policy_joint_order");
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "Holosoma action dim");
    require_matching_int(params, {"policy.num_obs", "num_obs"}, obs_dim, "Holosoma obs dim");
    require_permutation_one_of(params, {"robot.joint_map", "joint_index"}, motor_count, "Holosoma joint map");
    require_vector_one_of<float>(params, {"action.stiff_startup_pos", "stiff_startup_pos"},
                                 static_cast<size_t>(motor_count), "Holosoma stiff startup pos");
  } else if (variant == "protomotions") {
    validate_standard_robot_vectors(params, motor_count, false, false);
    require_matching_int(params, {"robot.motor_count"}, motor_count, "ProtoMotions motor count");
    require_positive_int(params, "reference.num_bodies");
    yamlc::required_vector<int>(params, "motion.future_step_indices");
  } else if (variant == "twist") {
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "TWIST action dim");
    require_matching_int(y, {"observation.policy_mimic_dim"}, protocol.twist_policy_mimic_dim, "TWIST policy mimic dim");
    require_matching_int(y, {"observation.proprio_dim"}, protocol.twist_proprio_dim, "TWIST proprio dim");
    require_matching_int(y, {"observation.obs_frame_dim"}, protocol.twist_obs_frame_dim, "TWIST obs frame dim");
    require_matching_int(y, {"observation.history_len"}, protocol.twist_history_len, "TWIST history len");
    require_matching_int(y, {"observation.input_dim"}, protocol.twist_input_dim, "TWIST input dim");
    require_matching_int(params, {"policy.num_obs", "num_obs"}, protocol.twist_input_dim, "TWIST input dim");
    require_vector_one_of<std::string>(params, {"robot.joint_order", "joint_order"},
                                       static_cast<size_t>(motor_count), "TWIST joint order");
    require_vector_one_of<float>(params, {"robot.joint_pos_lower_limit", "joint_pos_lower_limit"},
                                 static_cast<size_t>(motor_count), "TWIST lower limits");
    require_vector_one_of<float>(params, {"robot.joint_pos_upper_limit", "joint_pos_upper_limit"},
                                 static_cast<size_t>(motor_count), "TWIST upper limits");
    require_vector_one_of<float>(params, {"robot.torque_limits", "torque_limits"},
                                 static_cast<size_t>(motor_count), "TWIST torque limits");
    require_vector_one_of<int>(params, {"robot.body_motor_idx", "body_motor_idx"},
                               static_cast<size_t>(action_dim), "TWIST body motor indices");
    const int wrist_dim = protocol.twist_wrist_dim;
    require_vector_one_of<int>(params, {"robot.wrist_motor_idx", "wrist_motor_idx"},
                               static_cast<size_t>(wrist_dim), "TWIST wrist motor indices");
    require_vector_one_of<float>(params, {"robot.default_angles", "default_dof_pos"},
                                 static_cast<size_t>(action_dim), "TWIST default body angles");
    require_vector_one_of<float>(params, {"robot.body_kp", "body_kp"},
                                 static_cast<size_t>(action_dim), "TWIST body kp");
    require_vector_one_of<float>(params, {"robot.body_kd", "body_kd"},
                                 static_cast<size_t>(action_dim), "TWIST body kd");
    require_vector_one_of<float>(params, {"robot.wrist_default_angles", "wrist_default"},
                                 static_cast<size_t>(wrist_dim), "TWIST wrist default");
    require_vector_one_of<float>(params, {"motion.default_mimic_obs", "default_mimic_obs"},
                                 static_cast<size_t>(protocol.twist_mimic_dim),
                                 "TWIST default mimic obs");
  } else if (variant == "humanoid_gpt") {
    validate_standard_robot_vectors(params, motor_count, true, true);
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "Humanoid-GPT action dim");
    require_matching_int(params, {"policy.num_obs", "num_obs"}, obs_dim, "Humanoid-GPT obs dim");
    require_matching_int(params, {"reference.num_qpos", "num_qpos"},
                         protocol.humanoid_gpt_qpos_dim, "Humanoid-GPT qpos dim");
    require_matching_int(params, {"reference.num_qvel", "num_qvel"},
                         protocol.humanoid_gpt_qvel_dim, "Humanoid-GPT qvel dim");
    require_vector_one_of<std::string>(params, {"reference.keypoint_order", "keypoint_order"},
                                       static_cast<size_t>(protocol.humanoid_gpt_keypoint_count),
                                       "Humanoid-GPT keypoints");
  } else if (variant == "sonic") {
    validate_standard_robot_vectors(params, motor_count, true, true);
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "SONIC action dim");
    require_matching_int(params, {"policy.decoder_obs_dim"}, kSonicDecoderObsDim, "SONIC decoder obs dim");
    require_matching_int(params, {"policy.encoder_obs_dim"},
                         sonic_encoder_obs_dim_for_spec(y), "SONIC encoder obs dim");
    require_matching_int(params, {"policy.token_dim"}, kSonicTokenDim, "SONIC token dim");
    require_permutation_one_of(params, {"robot.isaaclab_to_hardware"}, motor_count, "SONIC IsaacLab to hardware map");
    require_permutation_one_of(params, {"robot.hardware_to_isaaclab"}, motor_count, "SONIC hardware to IsaacLab map");
  } else if (variant == "php") {
    validate_standard_robot_vectors(params, motor_count, true, true, "robot.policy_joint_order");
    require_permutation_one_of(params, {"robot.joint_map"}, motor_count, "PHP joint map");
    require_vector_one_of<float>(params, {"robot.joint_signs"}, static_cast<size_t>(motor_count), "PHP joint signs");
    require_vector_one_of<float>(params, {"robot.default_velocities"}, static_cast<size_t>(motor_count), "PHP default velocities");
    require_vector_one_of<float>(params, {"robot.startup_angles"}, static_cast<size_t>(motor_count), "PHP startup pose");
    require_vector_one_of<float>(params, {"robot.startup_kp"}, static_cast<size_t>(motor_count), "PHP startup kp");
    require_vector_one_of<float>(params, {"robot.startup_kd"}, static_cast<size_t>(motor_count), "PHP startup kd");
    if (yamlc::required_vector<int>(y, "observation.depth_input_shape") != std::vector<int>{1, 58, 87})
      throw std::runtime_error("PHP depth_input_shape must be [1,58,87]");
    const auto source = yamlc::required_scalar<std::string>(y, "sensor.depth_source");
    if (source != "shm" && source != "zero" && source != "realsense") throw std::runtime_error("PHP depth source must be realsense, shm or zero");
    if (source == "realsense") {
      yamlc::required_scalar<int>(y, "sensor.rs_width");
      yamlc::required_scalar<int>(y, "sensor.rs_height");
      yamlc::required_scalar<int>(y, "sensor.rs_fps");
      if (!yamlc::required_scalar<bool>(y, "sensor.require_depth_camera") ||
          !yamlc::required_scalar<bool>(y, "sensor.require_torso_imu") ||
          yamlc::required_scalar<bool>(y, "sensor.disable_torso_imu"))
        throw std::runtime_error("PHP hardware requires RealSense and torso IMU before ready");
    }
    yamlc::required_scalar<float>(y, "safety.kp_max");
    yamlc::required_scalar<float>(y, "safety.kd_max");
    const auto command_limits = yamlc::required_scalar<std::string>(y, "safety.command_limit_source");
    if (command_limits != "joint_position" && command_limits != "simulation_model_action" && command_limits != "pd_torque")
      throw std::runtime_error("PHP command_limit_source must be joint_position, pd_torque or simulation_model_action");
    if (command_limits == "simulation_model_action" && yamlc::required_scalar<std::string>(y, "runtime.net") != "lo")
      throw std::runtime_error("PHP simulation command limits require runtime.net=lo");
    require_matching_int(params, {"depth.height"}, 58, "PHP depth height");
    require_matching_int(params, {"depth.width"}, 87, "PHP depth width");
  } else if (variant == "instinct_parkour" || variant == "instinct") {
    require_matching_int(params, {"robot.motor_count"}, motor_count, "Instinct motor count");
    validate_standard_robot_vectors(params, motor_count, true, true, "robot.policy_joint_order");
    require_permutation_one_of(params, {"robot.joint_map"}, motor_count, "Instinct joint map");
    require_vector_one_of<float>(params, {"robot.joint_signs"}, static_cast<size_t>(motor_count), "Instinct joint signs");
    require_vector_one_of<float>(params, {"robot.action_offset"}, static_cast<size_t>(motor_count), "Instinct action offset");
  } else if (variant == "steadytray") {
    validate_standard_robot_vectors(params, motor_count, true, true, "robot.policy_joint_order");
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "SteadyTray action dim");
    require_matching_int(params, {"policy.num_obs", "num_obs"},
                         yamlc::required_scalar<int>(y, "observation.frame_dim"), "SteadyTray frame dim");
    require_permutation_one_of(params, {"robot.joint_map", "policy_to_xml"}, motor_count, "SteadyTray joint map");
    require_permutation_one_of(params, {"robot.robot_to_policy", "xml_to_policy"}, motor_count, "SteadyTray robot_to_policy");
  } else if (variant == "omnixtreme") {
    validate_standard_robot_vectors(params, motor_count, true, true);
    require_matching_int(params, {"policy.num_actions", "num_actions"}, action_dim, "OmniXtreme action dim");
    require_permutation_one_of(params, {"robot.joint_permutation"}, motor_count, "OmniXtreme joint_permutation");
    require_permutation_one_of(params, {"robot.inverse_joint_permutation"}, motor_count, "OmniXtreme inverse_joint_permutation");
    require_positive_int(params, "motion.anchor_body_index");
  } else {
    throw std::runtime_error("schema checker has no params rules for variant: " + variant);
  }
}

GenericPlayerKind expected_player_for_spec(const PolicyContract& contract) {
  return contract.player_kind;
}

void check_spec(const fs::path& raw_spec, bool strict_onnx) {
  const fs::path spec = fs::absolute(raw_spec).lexically_normal();
  if (!fs::exists(spec)) throw std::runtime_error("spec not found: " + spec.string());
  const YAML::Node y = YAML::LoadFile(spec.string());
  const std::string tmpl = lower(yamlc::required_scalar<std::string>(y, "runtime.template"));
  const std::string variant = lower(yamlc::required_scalar<std::string>(y, "runtime.variant"));
  const PolicyContract* contract = contract_for_template(tmpl);
  if (!contract) {
    throw std::runtime_error("unsupported runtime.template: " + tmpl);
  }
  if (!variant_allowed_for_template(tmpl, variant)) {
    throw std::runtime_error("runtime.variant " + variant + " is not valid for " + tmpl +
                             "; supported variants: " + format_supported_variants_for_template(tmpl));
  }
  const auto* variant_info = variant_contract(variant);
  const PolicyProtocolConstraint* protocol = protocol_constraint_for_variant(variant);
  if (!protocol) {
    throw std::runtime_error("missing PolicyProtocolConstraint for variant: " + variant);
  }

  const int motor_count = require_positive_int(y, "robot.motor_count");
  const int obs_dim = require_positive_int(y, "observation.obs_dim");
  const int action_dim = require_positive_int(y, "action.action_dim");
  if (protocol->motor_count > 0 && motor_count != protocol->motor_count) {
    throw std::runtime_error("robot.motor_count protocol mismatch: expected " +
                             std::to_string(protocol->motor_count) + ", got " + std::to_string(motor_count));
  }
  if (protocol->obs_dim > 0 && obs_dim != protocol->obs_dim) {
    throw std::runtime_error("observation.obs_dim protocol mismatch: expected " +
                             std::to_string(protocol->obs_dim) + ", got " + std::to_string(obs_dim));
  }
  if (protocol->action_dim > 0 && action_dim != protocol->action_dim) {
    throw std::runtime_error("action.action_dim protocol mismatch: expected " +
                             std::to_string(protocol->action_dim) + ", got " + std::to_string(action_dim));
  }
  validate_protocol_yaml_dim_rules(y, variant);
  require_positive_int(y, "runtime.inference_hz");
  require_positive_int(y, "runtime.publish_hz");
  require_positive_int(y, "runtime.onnx_threads");
  yamlc::required_scalar<std::string>(y, "handoff.cmd_topic");
  yamlc::required_scalar<std::string>(y, "handoff.state_topic");
  require_positive_int(y, "handoff.mode_machine");
  yamlc::required_scalar<bool>(y, "handoff.damping_on_stop");
  const YAML::Node handoff = y["handoff"];
  const double activation_blend_sec = handoff && handoff["activation_blend_sec"]
      ? handoff["activation_blend_sec"].as<double>()
      : 0.10;
  if (!std::isfinite(activation_blend_sec) || activation_blend_sec < 0.0 ||
      activation_blend_sec > 5.0) {
    throw std::runtime_error(
        "handoff.activation_blend_sec must be finite and within [0, 5]");
  }

  const fs::path params = require_existing_path(spec, yamlc::required_scalar<std::string>(y, "robot.params"), "robot.params");
  validate_model(spec, y, strict_onnx, *contract, *protocol);
  validate_source_sections(spec, y, *contract, *protocol);
  if (variant == "pbfm") {
    action_go::runtime::PbfmSpec::load(y, spec);
    if (strict_onnx) {
      const auto model = resolve_relative(spec, yamlc::required_scalar<std::string>(y, "model.path"));
      if (fs::exists(model)) {
        OnnxRunner runner;
        runner.load(model, 1);
        action_go::runtime::PbfmPolicy::validate_model(runner);
      }
    }
  }
  validate_params(y, YAML::LoadFile(params.string()), variant, motor_count, obs_dim, action_dim, *protocol);

  std::cout << "[schema] OK " << spec.string() << "\n"
            << "  template=" << contract->template_name
            << " variant=" << variant
            << " player=" << player_binary_name(expected_player_for_spec(*contract))
            << " category=\"" << contract->category_name << "\""
            << " template_core=" << (contract->core_paper_coverage ? "true" : "false")
            << " variant_core=" << (variant_info && variant_info->core_paper_coverage ? "true" : "false") << "\n"
            << "  motor_count=" << motor_count
            << " obs_dim=" << obs_dim
            << " action_dim=" << action_dim
            << " params=" << params.string() << "\n";
  std::cout << "  protocol=" << format_protocol_constraint_summary(*protocol) << "\n";
  if (protocol->optional_build) {
    std::cout << "  note=" << protocol->optional_build_note << "\n";
  }
}

std::vector<fs::path> find_all_specs() {
  std::vector<fs::path> specs;
  const fs::path actions = fs::current_path() / "actions";
  if (!fs::exists(actions)) throw std::runtime_error("actions/ not found from cwd: " + fs::current_path().string());
  for (const fs::directory_entry& ent : fs::recursive_directory_iterator(actions)) {
    if (!ent.is_regular_file()) continue;
    const fs::path p = ent.path();
    const std::string name = p.filename().string();
    if (name.rfind("generic_", 0) == 0 && (p.extension() == ".yaml" || p.extension() == ".yml")) {
      specs.push_back(p);
    }
  }
  std::sort(specs.begin(), specs.end());
  return specs;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    g_allow_missing_assets = args.allow_missing_assets;
    std::vector<fs::path> specs = args.specs;
    if (args.all) {
      const auto all_specs = find_all_specs();
      specs.insert(specs.end(), all_specs.begin(), all_specs.end());
    }
    std::sort(specs.begin(), specs.end());
    specs.erase(std::unique(specs.begin(), specs.end()), specs.end());

    int failures = 0;
    for (const fs::path& spec : specs) {
      try {
        check_spec(spec, args.strict_onnx);
      } catch (const std::exception& e) {
        ++failures;
        std::cerr << "[schema] FAIL " << spec.string() << ": " << e.what() << "\n";
      }
    }
    if (failures != 0) {
      std::cerr << "[schema] " << failures << " spec(s) failed\n";
      return 1;
    }
    std::cout << "[schema] checked " << specs.size() << " spec(s)"
              << ", missing_assets=" << g_missing_assets
              << (args.installed_only ? " mode=installed-only" :
                  args.allow_missing_assets ? " mode=allow-missing-assets" : " mode=strict")
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[schema] ERROR: " << e.what() << "\n";
    return 1;
  }
}
