#include "action_go/runtime/policy_contract.h"
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

int run_omnixtreme_runtime(int argc, char** argv);

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
      std::cout << "Usage: generic_composite_policy_player --spec actions/.../generic_*.yaml [runtime overrides]\n"
                << "  Templates: " << format_supported_templates(GenericPlayerKind::kComposite) << ".\n"
                << "  Variants: " << format_supported_variants(GenericPlayerKind::kComposite) << ".\n"
                << "  --event-log PATH writes JSONL runtime evidence events.\n";
      std::exit(0);
    } else {
      cli.passthrough.push_back(k);
      const bool takes_value =
          k == "--net" || k == "--domain-id" || k == "--model" || k == "--base-model" ||
          k == "--residual-model" || k == "--fk-model" || k == "--params" ||
          k == "--trajectory" || k == "--motion" || k == "--cmd-topic" ||
          k == "--state-topic" || k == "--ready-file" || k == "--start-file" ||
          k == "--preview-file" || k == "--commit-file" ||
          k == "--discard-file" || k == "--preview-timeout-sec" ||
          k == "--ep" || k == "--execution-provider" || k == "--gpu-device-id" ||
          k == "--device-id" || k == "--trt-cache-dir" || k == "--trt-engine-cache-dir" ||
          k == "--trt-workspace-size" || k == "--hz" || k == "--publish-hz" ||
          k == "--onnx-threads" || k == "--mode-machine" || k == "--motion-start-frame" ||
          k == "--motion-start-step" || k == "--max-frames" || k == "--max-steps" ||
          k == "--duration-sec" || k == "--startup-interp-sec" ||
          k == "--target-interp-sec" || k == "--residual-gain" ||
          k == "--initial-noise-scale" || k == "--stats-interval" ||
          k == "--warmup-iters" || k == "--prepare-ramp-sec" || k == "--prepare-hold-sec";
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

void add_bool_inverse_flag(std::vector<std::string>& out, const std::string& false_flag,
                           const YAML::Node& node, const char* field) {
  if (node && node[field] && !node[field].as<bool>()) out.push_back(false_flag);
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
  std::cout << "[generic_composite_policy_player] final runtime args:";
  for (const std::string& arg : args) std::cout << ' ' << arg;
  std::cout << "\n";
}

const PolicyContract* validate_contract(const std::string& tmpl,
                                        const std::string& variant,
                                        const fs::path& spec) {
  const auto player = GenericPlayerKind::kComposite;
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

int run_with_args(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (std::string& s : args) argv.push_back(const_cast<char*>(s.c_str()));
  const int argc = static_cast<int>(argv.size());
  argv.push_back(nullptr);
  return run_omnixtreme_runtime(argc, argv.data());
}

}  // namespace

int main(int argc, char** argv) {
  RuntimeEventLogger event_log;
  std::string tmpl;
  std::string variant;
  try {
    const Cli cli = parse_cli(argc, argv);
    event_log.open(cli.event_log);
    event_log.log("process_start",
                  {event_string("player", "generic_composite_policy_player"),
                   event_string("spec", cli.spec)});
    const fs::path spec = fs::absolute(fs::path(cli.spec)).lexically_normal();
    const YAML::Node y = YAML::LoadFile(spec.string());
    const YAML::Node runtime = y["runtime"];
    tmpl = lower(scalar_or<std::string>(runtime, "template", ""));
    variant = lower(scalar_or<std::string>(runtime, "variant", ""));
    const PolicyContract* contract = validate_contract(tmpl, variant, spec);
    if (!contract || std::string(contract->template_name) != "composite_motion_policy") {
      throw std::runtime_error("unsupported composite policy template/variant in " + spec.string());
    }
    event_log.log("spec_loaded",
                  {event_string("player", "generic_composite_policy_player"),
                   event_string("template", contract->template_name),
                   event_string("variant", variant),
                   event_string("spec", spec.string()),
                   event_int("inference_hz", scalar_or<int>(runtime, "inference_hz", 0)),
                   event_int("publish_hz", scalar_or<int>(runtime, "publish_hz", 0))});

    const YAML::Node robot = y["robot"];
    const YAML::Node model = y["model"];
    const YAML::Node motion = y["motion"];
    const YAML::Node handoff = y["handoff"];
    const YAML::Node trt = y["tensorrt"];
    const YAML::Node action = y["action"];
    require_node(robot, "robot");
    require_node(model, "model");
    require_node(motion, "motion");
    require_existing_path(spec, require_string(model, "base", "model"), "model.base");
    require_existing_path(spec, require_string(model, "residual", "model"), "model.residual");
    require_existing_path(spec, require_string(model, "fk", "model"), "model.fk");
    require_existing_path(spec, require_string(robot, "params", "robot"), "robot.params");
    require_existing_path(spec, require_string(motion, "path", "motion"), "motion.path");
    std::vector<std::string> args{"generic_composite_policy_player"};
    add(args, "--net", scalar_or<std::string>(runtime, "net", ""));
    add_scalar<int>(args, "--domain-id", runtime, "domain_id");
    add_path(args, "--model", spec, scalar_or<std::string>(model, "base", scalar_or<std::string>(model, "path", "")));
    add_path(args, "--residual-model", spec, scalar_or<std::string>(model, "residual", ""));
    add_path(args, "--fk-model", spec, scalar_or<std::string>(model, "fk", ""));
    add_path(args, "--params", spec, scalar_or<std::string>(robot, "params", ""));
    add_path(args, "--trajectory", spec, scalar_or<std::string>(motion, "path", ""));
    add(args, "--cmd-topic", scalar_or<std::string>(handoff, "cmd_topic", ""));
    add(args, "--state-topic", scalar_or<std::string>(handoff, "state_topic", ""));
    add(args, "--ready-file", scalar_or<std::string>(handoff, "ready_file", ""));
    add(args, "--start-file", scalar_or<std::string>(handoff, "start_file", ""));
    add(args, "--ep", scalar_or<std::string>(runtime, "execution_provider", ""));
    add_path(args, "--trt-cache-dir", spec, scalar_or<std::string>(trt, "cache_dir", ""));
    add_scalar<int>(args, "--gpu-device-id", trt, "gpu_device_id");
    add_scalar<unsigned long long>(args, "--trt-workspace-size", trt, "workspace_size");
    add_scalar<int>(args, "--hz", runtime, "inference_hz");
    add_scalar<int>(args, "--publish-hz", runtime, "publish_hz");
    add_scalar<int>(args, "--onnx-threads", runtime, "onnx_threads");
    add_scalar<int>(args, "--mode-machine", handoff, "mode_machine");
    add_scalar<int>(args, "--motion-start-frame", motion, "start_frame");
    add_scalar<int>(args, "--max-frames", motion, "max_frames");
    add_scalar<double>(args, "--duration-sec", runtime, "duration_sec");
    add_scalar<double>(args, "--startup-interp-sec", runtime, "startup_interp_sec");
    add_scalar<double>(args, "--target-interp-sec", runtime, "target_interp_sec");
    add_damping_on_stop(args, handoff);
    add_scalar<double>(args, "--residual-gain", action, "residual_gain");
    add_scalar<double>(args, "--initial-noise-scale", action, "initial_noise_scale");
    add_scalar<int>(args, "--stats-interval", runtime, "stats_interval");
    add_scalar<int>(args, "--warmup-iters", runtime, "warmup_iters");
    add_scalar<double>(args, "--prepare-ramp-sec", runtime, "prepare_ramp_sec");
    add_scalar<double>(args, "--prepare-hold-sec", runtime, "prepare_hold_sec");
    add_bool_flag(args, "--trt-fp16", trt, "fp16");
    add_bool_inverse_flag(args, "--no-trt-fp16", trt, "fp16");
    add_bool_flag(args, "--joint-limit-clamp", action, "joint_limit_clamp");
    add_bool_inverse_flag(args, "--no-friction-feedforward", action, "friction_feedforward");
    append_legacy_args(args, y);
    if (!cli.event_log.empty()) {
      args.push_back("--event-log");
      args.push_back(cli.event_log);
    }
    args.insert(args.end(), cli.passthrough.begin(), cli.passthrough.end());
    print_args_if_check_only(args, cli.check_only);
    const int rc = run_with_args(std::move(args));
    event_log.log("process_exit",
                  {event_string("player", "generic_composite_policy_player"),
                   event_string("template", contract->template_name),
                   event_string("variant", variant),
                   event_bool("ok", rc == 0),
                   event_string("reason", rc == 0 ? "ok" : "runtime_returned_error")});
    return rc;
  } catch (const std::exception& e) {
    event_log.log("process_exit",
                  {event_string("player", "generic_composite_policy_player"),
                   event_string("template", tmpl),
                   event_string("variant", variant),
                   event_bool("ok", false),
                   event_string("reason", e.what())});
    std::cerr << "[generic_composite_policy_player] ERROR: " << e.what() << "\n";
    return 1;
  }
}
