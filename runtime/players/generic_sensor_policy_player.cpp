#include "action_go/runtime/policy_contract.h"
#include "action_go/runtime/generic_policy_runtime.h"
#include "action_go/runtime/event_logger.h"

#include <yaml-cpp/yaml.h>

#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using action_go::runtime::GenericPlayerKind;
using action_go::runtime::PolicyContract;
using action_go::runtime::contract_for_template;
using action_go::runtime::contract_for_variant;
using action_go::runtime::event_bool;
using action_go::runtime::event_int;
using action_go::runtime::event_string;
using action_go::runtime::format_supported_templates;
using action_go::runtime::format_supported_variants;
using action_go::runtime::format_supported_variants_for_template;
using action_go::runtime::player_binary_name;
using action_go::runtime::RuntimeEventLogger;
using action_go::runtime::variant_allowed_for_template;

int run_instinct_parkour_runtime(int argc, char** argv);
int run_steadytray_runtime(int argc, char** argv);

namespace {

struct Cli {
  std::string spec;
  std::string event_log;
  std::vector<std::string> passthrough;
  bool check_only{false};
};

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

Cli parse_cli(int argc, char** argv) {
  Cli cli;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k == "--spec") {
      if (i + 1 >= argc) throw std::runtime_error("missing value for --spec");
      cli.spec = argv[++i];
    } else if (k == "--event-log") {
      if (i + 1 >= argc) throw std::runtime_error("missing value for --event-log");
      cli.event_log = argv[++i];
    } else if (k == "--check-only") {
      cli.check_only = true;
      cli.passthrough.push_back(k);
    } else if (k == "--help") {
      std::cout << "Usage: generic_sensor_policy_player --spec actions/.../generic_*.yaml [runtime overrides]\n"
                << "  Templates: " << format_supported_templates(GenericPlayerKind::kSensor) << ".\n"
                << "  Variants: " << format_supported_variants(GenericPlayerKind::kSensor) << ".\n"
                << "  --event-log PATH writes JSONL runtime evidence events.\n"
                << "  Extra CLI args override spec-derived legacy runtime args.\n";
      std::exit(0);
    } else {
      cli.passthrough.push_back(k);
      const bool takes_value =
          k == "--net" || k == "--cmd-topic" || k == "--state-topic" ||
          k == "--ready-file" || k == "--start-file" ||
          k == "--trace-file" || k == "--depth-shm-name" ||
          k == "--preview-file" || k == "--commit-file" ||
          k == "--depth-model" || k == "--actor-model" || k == "--variant" ||
          k == "--trace-file" || k == "--command-file" || k == "--command-index" ||
          k == "--command-source" || k == "--depth-shm-name" || k == "--command-limit-source" ||
          k == "--kp-max" || k == "--kd-max" ||
          k == "--discard-file" || k == "--preview-timeout-sec" || k == "--hz" ||
          k == "--publish-hz" || k == "--onnx-threads" || k == "--duration-sec" ||
          k == "--object-source" || k == "--depth-source" || k == "--tag-size" ||
          k == "--tag-id" || k == "--udp-port" || k == "--stats-interval" ||
          k == "--target-interp-sec" || k == "--target-rate-limit" ||
          k == "--startup-sec" || k == "--startup-hold-sec" ||
          k == "--cold-start-hold-sec" || k == "--depth-camera-timeout-sec" ||
          k == "--min-depth-camera-frames" || k == "--torso-imu-timeout-sec" ||
          k == "--camera-timeout-sec" || k == "--pose-timeout-sec" ||
          k == "--warmup-steps" || k == "--model" || k == "--params" ||
          k == "--activation-blend-sec" ||
          k == "--logdir" || k == "--standdir" || k == "--stand-logdir" ||
          k == "--rs-width" || k == "--rs-height" || k == "--rs-fps" ||
          k == "--rs-resolution" || k == "--mode-machine" || k == "--cmd" ||
          k == "--cmd-init" || k == "--cmd-vx" || k == "--cmd-vy" ||
          k == "--cmd-yaw" || k == "--vision-to-policy-pos" ||
          k == "--vision-to-policy-quat" || k == "--tag-to-object-pos" ||
          k == "--tag-to-object-quat";
      if (takes_value && i + 1 < argc) cli.passthrough.push_back(argv[++i]);
    }
  }
  if (cli.spec.empty()) throw std::runtime_error("--spec is required");
  return cli;
}

template <typename T>
T scalar_or(const YAML::Node& node, const char* key, const T& fallback) {
  if (!node || !node[key]) return fallback;
  return node[key].as<T>();
}

fs::path resolve_relative(const fs::path& spec, const std::string& value) {
  if (value.empty()) return {};
  fs::path p(value);
  if (p.is_absolute()) return p.lexically_normal();
  return (spec.parent_path() / p).lexically_normal();
}

void require_node(const YAML::Node& node, const std::string& name) {
  if (!node) throw std::runtime_error("spec missing required section: " + name);
}

std::string require_string(const YAML::Node& node, const char* key, const std::string& scope) {
  if (!node || !node[key]) throw std::runtime_error("spec missing required field: " + scope + "." + key);
  const std::string value = node[key].as<std::string>();
  if (value.empty()) throw std::runtime_error("spec field is empty: " + scope + "." + key);
  return value;
}

fs::path require_existing_path(const fs::path& spec,
                               const std::string& value,
                               const std::string& field) {
  const fs::path path = resolve_relative(spec, value);
  if (path.empty()) throw std::runtime_error("spec field is empty: " + field);
  if (!fs::exists(path)) throw std::runtime_error("spec path does not exist for " + field + ": " + path.string());
  return path;
}

void add(std::vector<std::string>& out, const std::string& key, const std::string& value) {
  if (value.empty()) return;
  out.push_back(key);
  out.push_back(value);
}

void add_path(std::vector<std::string>& out, const std::string& key,
              const fs::path& spec, const std::string& value) {
  if (value.empty()) return;
  add(out, key, resolve_relative(spec, value).string());
}

template <typename T>
void add_scalar(std::vector<std::string>& out, const std::string& key,
                const YAML::Node& node, const char* field) {
  if (!node || !node[field]) return;
  add(out, key, std::to_string(node[field].as<T>()));
}

void add_bool_flag(std::vector<std::string>& out, const std::string& flag,
                   const YAML::Node& node, const char* field) {
  if (node && node[field] && node[field].as<bool>()) out.push_back(flag);
}

void add_damping_on_stop(std::vector<std::string>& out, const YAML::Node& handoff) {
  if (!handoff || !handoff["damping_on_stop"]) return;
  out.push_back(handoff["damping_on_stop"].as<bool>() ? "--damping-on-stop" : "--no-damping-on-stop");
}

void append_legacy_args(std::vector<std::string>& out, const YAML::Node& node) {
  if (!node || !node["legacy_args"]) return;
  for (const auto& v : node["legacy_args"]) out.push_back(v.as<std::string>());
}

void print_args_if_check_only(const std::vector<std::string>& args, bool check_only) {
  if (!check_only) return;
  std::cout << "[generic_sensor_policy_player] final runtime args:";
  for (const std::string& arg : args) std::cout << ' ' << arg;
  std::cout << "\n";
}

const PolicyContract* validate_contract(const std::string& tmpl,
                                        const std::string& variant,
                                        const fs::path& spec) {
  const auto player = GenericPlayerKind::kSensor;
  if (!tmpl.empty()) {
    const PolicyContract* contract = contract_for_template(tmpl);
    if (!contract) {
      throw std::runtime_error("unsupported runtime.template for " +
                               player_binary_name(player) + ": " + tmpl +
                               ". Supported templates: " +
                               format_supported_templates(player));
    }
    if (contract->player_kind != player) {
      throw std::runtime_error("runtime.template " + tmpl +
                               " belongs to " + player_binary_name(contract->player_kind) +
                               "; " + player_binary_name(player) +
                               " supports only " + format_supported_templates(player));
    }
    if (!variant.empty() && !variant_allowed_for_template(tmpl, variant)) {
      throw std::runtime_error(std::string(contract->template_name) +
                               " supports variants: " +
                               format_supported_variants_for_template(tmpl));
    }
    return contract;
  }
  if (!variant.empty()) {
    const PolicyContract* contract = contract_for_variant(variant);
    if (!contract || contract->player_kind != player) {
      throw std::runtime_error("unsupported runtime.variant for " +
                               player_binary_name(player) + ": " + variant +
                               ". Supported variants: " +
                               format_supported_variants(player));
    }
    return contract;
  }
  throw std::runtime_error("spec runtime.template or runtime.variant is required in " + spec.string());
}

int run_with_args(int (*fn)(int, char**), std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& s : args) argv.push_back(const_cast<char*>(s.c_str()));
  const int argc = static_cast<int>(argv.size());
  argv.push_back(nullptr);
  return fn(argc, argv.data());
}

std::vector<std::string> common_args(const fs::path& spec, const YAML::Node& y) {
  const YAML::Node runtime = y["runtime"];
  const YAML::Node handoff = y["handoff"];
  std::vector<std::string> args{"generic_sensor_policy_player"};
  add(args, "--net", scalar_or<std::string>(runtime, "net", ""));
  add(args, "--cmd-topic", scalar_or<std::string>(handoff, "cmd_topic", ""));
  add(args, "--state-topic", scalar_or<std::string>(handoff, "state_topic", ""));
  add(args, "--ready-file", scalar_or<std::string>(handoff, "ready_file", ""));
  add(args, "--start-file", scalar_or<std::string>(handoff, "start_file", ""));
  add_scalar<int>(args, "--hz", runtime, "inference_hz");
  add_scalar<int>(args, "--publish-hz", runtime, "publish_hz");
  add_scalar<int>(args, "--onnx-threads", runtime, "onnx_threads");
  add_scalar<int>(args, "--stats-interval", runtime, "stats_interval");
  add_scalar<int>(args, "--mode-machine", handoff, "mode_machine");
  add_scalar<double>(args, "--activation-blend-sec", handoff, "activation_blend_sec");
  add_scalar<double>(args, "--duration-sec", runtime, "duration_sec");
  add_damping_on_stop(args, handoff);
  (void)spec;
  return args;
}

int run_instinct(const fs::path& spec, const YAML::Node& y,
                 const std::vector<std::string>& passthrough,
                 bool check_only) {
  const YAML::Node robot = y["robot"];
  const YAML::Node model = y["model"];
  const YAML::Node sensor = y["sensor"];
  const YAML::Node startup = y["startup"];
  require_node(robot, "robot");
  require_node(model, "model");
  require_node(sensor, "sensor");
  require_existing_path(spec, require_string(robot, "params", "robot"), "robot.params");
  require_existing_path(spec, require_string(model, "logdir", "model"), "model.logdir");
  if (model["standdir"]) {
    require_existing_path(spec, model["standdir"].as<std::string>(), "model.standdir");
  }
  require_string(sensor, "depth_source", "sensor");
  auto args = common_args(spec, y);
  if (scalar_or<std::string>(y["runtime"], "variant", "") == "php") {
    add(args, "--variant", "php");
    add_path(args, "--params", spec, robot["params"].as<std::string>());
    add_path(args, "--depth-model", spec, model["sessions"]["depth_encoder"]["path"].as<std::string>());
    add_path(args, "--actor-model", spec, model["sessions"]["actor"]["path"].as<std::string>());
    if (y["action"]["clip_by_torque"] && !y["action"]["clip_by_torque"].as<bool>()) args.push_back("--no-clip-by-torque");
    add_scalar<float>(args, "--kp-max", y["safety"], "kp_max");
    add(args, "--command-limit-source", require_string(y["safety"], "command_limit_source", "safety"));
    add_scalar<float>(args, "--kd-max", y["safety"], "kd_max");
    add(args, "--depth-shm-name", scalar_or<std::string>(sensor, "shm_name", "depth_img_shm"));
  }
  add_path(args, "--logdir", spec, scalar_or<std::string>(model, "logdir", ".."));
  add_path(args, "--standdir", spec, scalar_or<std::string>(model, "standdir", ""));
  add(args, "--depth-source", scalar_or<std::string>(sensor, "depth_source", ""));
  add_scalar<int>(args, "--rs-width", sensor, "rs_width");
  add_scalar<int>(args, "--rs-height", sensor, "rs_height");
  add_scalar<int>(args, "--rs-fps", sensor, "rs_fps");
  add_scalar<int>(args, "--min-depth-camera-frames", sensor, "min_depth_camera_frames");
  add_scalar<double>(args, "--depth-camera-timeout-sec", sensor, "depth_camera_timeout_sec");
  add_scalar<double>(args, "--torso-imu-timeout-sec", sensor, "torso_imu_timeout_sec");
  add_bool_flag(args, "--require-depth-camera", sensor, "require_depth_camera");
  add_bool_flag(args, "--require-torso-imu", sensor, "require_torso_imu");
  add_bool_flag(args, "--no-torso-imu", sensor, "disable_torso_imu");
  add_scalar<double>(args, "--target-interp-sec", startup, "target_interp_sec");
  add_scalar<double>(args, "--target-rate-limit", startup, "target_rate_limit");
  add_scalar<double>(args, "--startup-sec", startup, "startup_sec");
  add_scalar<double>(args, "--startup-hold-sec", startup, "startup_hold_sec");
  add_scalar<double>(args, "--cold-start-hold-sec", startup, "cold_start_hold_sec");
  add_bool_flag(args, "--cold-start-before-policy", startup, "cold_start_before_policy");
  if (startup && startup["use_normalizer"] && !startup["use_normalizer"].as<bool>()) {
    args.push_back("--no-use-normalizer");
  }
  append_legacy_args(args, y);
  args.insert(args.end(), passthrough.begin(), passthrough.end());
  print_args_if_check_only(args, check_only);
  return run_with_args(run_instinct_parkour_runtime, std::move(args));
}

int run_steadytray(const fs::path& spec, const YAML::Node& y,
                   const std::vector<std::string>& passthrough,
                   bool check_only) {
  const YAML::Node model = y["model"];
  const YAML::Node robot = y["robot"];
  const YAML::Node object = y["object"];
  require_node(model, "model");
  require_node(robot, "robot");
  require_node(object, "object");
  require_existing_path(spec, require_string(model, "path", "model"), "model.path");
  require_existing_path(spec, require_string(robot, "params", "robot"), "robot.params");
  require_string(object, "source", "object");
  if (lower(object["source"].as<std::string>()) == "apriltag" && !object["tag_size"]) {
    throw std::runtime_error("spec missing required field for apriltag source: object.tag_size");
  }
  auto args = common_args(spec, y);
  add_path(args, "--model", spec, scalar_or<std::string>(model, "path", ""));
  add_path(args, "--params", spec, scalar_or<std::string>(robot, "params", scalar_or<std::string>(model, "params", "")));
  add(args, "--object-source", scalar_or<std::string>(object, "source", ""));
  add_scalar<int>(args, "--udp-port", object, "udp_port");
  add_scalar<int>(args, "--tag-id", object, "tag_id");
  add_scalar<float>(args, "--tag-size", object, "tag_size");
  add_scalar<double>(args, "--camera-timeout-sec", object, "camera_timeout_sec");
  add_scalar<double>(args, "--pose-timeout-sec", object, "pose_timeout_sec");
  add_scalar<int>(args, "--warmup-steps", y["runtime"], "warmup_steps");
  add_scalar<double>(args, "--target-interp-sec", y["runtime"], "target_interp_sec");
  add_bool_flag(args, "--require-tag", object, "require_tag");
  if (object && object["exit_on_tag_lost"] && !object["exit_on_tag_lost"].as<bool>()) {
    args.push_back("--allow-tag-lost");
  }
  append_legacy_args(args, y);
  args.insert(args.end(), passthrough.begin(), passthrough.end());
  print_args_if_check_only(args, check_only);
  return run_with_args(run_steadytray_runtime, std::move(args));
}

}  // namespace

int main(int argc, char** argv) {
  RuntimeEventLogger event_log;
  std::string tmpl;
  std::string variant;
  try {
#ifdef ACTION_GO_WITH_REALSENSE
    constexpr const char* kRealSenseBuildFeature = "enabled";
#else
    constexpr const char* kRealSenseBuildFeature = "disabled";
#endif
#ifdef ACTION_GO_WITH_APRILTAG
    constexpr const char* kAprilTagBuildFeature = "enabled";
#else
    constexpr const char* kAprilTagBuildFeature = "disabled";
#endif
    std::cout << "[generic_sensor_policy_player] build features: realsense="
              << kRealSenseBuildFeature << " apriltag=" << kAprilTagBuildFeature
              << "\n";
    const Cli cli = parse_cli(argc, argv);
    event_log.open(cli.event_log);
    event_log.log("process_start",
                  {event_string("player", "generic_sensor_policy_player"),
                   event_string("spec", cli.spec)});
    const fs::path spec = fs::absolute(fs::path(cli.spec)).lexically_normal();
    const YAML::Node y = YAML::LoadFile(spec.string());
    tmpl = lower(scalar_or<std::string>(y["runtime"], "template", ""));
    variant = lower(scalar_or<std::string>(y["runtime"], "variant", ""));
    const PolicyContract* contract = validate_contract(tmpl, variant, spec);
    if (variant == "pbfm") {
      return action_go::runtime::run_generic_policy_runtime(
          argc, argv, action_go::runtime::GenericPolicyRuntimeScope::kSensor);
    }
    const std::string canonical_template = contract ? contract->template_name : "";
    event_log.log("spec_loaded",
                  {event_string("player", "generic_sensor_policy_player"),
                   event_string("template", canonical_template),
                   event_string("variant", variant),
                   event_string("spec", spec.string()),
                   event_int("inference_hz", scalar_or<int>(y["runtime"], "inference_hz", 0)),
                   event_int("publish_hz", scalar_or<int>(y["runtime"], "publish_hz", 0))});
    std::vector<std::string> passthrough = cli.passthrough;
    if (!cli.event_log.empty()) {
      passthrough.push_back("--event-log");
      passthrough.push_back(cli.event_log);
    }
    int rc = 1;
    if (canonical_template == "depth_conditioned_actor") {
      rc = run_instinct(spec, y, passthrough, cli.check_only);
    } else if (canonical_template == "object_conditioned_actor") {
      rc = run_steadytray(spec, y, passthrough, cli.check_only);
    } else {
      throw std::runtime_error("unsupported sensor policy template/variant in " + spec.string());
    }
    event_log.log("process_exit",
                  {event_string("player", "generic_sensor_policy_player"),
                   event_string("template", canonical_template),
                   event_string("variant", variant),
                   event_bool("ok", rc == 0),
                   event_string("reason", rc == 0 ? "ok" : "runtime_returned_error")});
    return rc;
  } catch (const std::exception& e) {
    event_log.log("process_exit",
                  {event_string("player", "generic_sensor_policy_player"),
                   event_string("template", tmpl),
                   event_string("variant", variant),
                   event_bool("ok", false),
                   event_string("reason", e.what())});
    std::cerr << "[generic_sensor_policy_player] ERROR: " << e.what() << "\n";
    return 1;
  }
}
