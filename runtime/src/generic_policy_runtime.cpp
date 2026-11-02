#include "generic_policy_runtime_internal.h"

namespace action_go::runtime::generic_policy_runtime {

GenericPolicyPlayer::GenericPolicyPlayer(Args args) : args_(std::move(args)) {
    event_log_.open(args_.event_log);
    event_log_.log("process_start",
                   {event_string("player", player_label()),
                    event_string("spec", args_.spec),
                    event_string("scope", player_kind() == GenericPlayerKind::kSensor ? "sensor" :
                                           is_motion_tracking_player() ? "motion_tracking" : "policy")});
  }

GenericPolicyPlayer::~GenericPolicyPlayer() {
    pbfm_realsense_.reset();
    subscriber_.reset();
    publisher_.reset();
  }

void GenericPolicyPlayer::run() {
    load_spec();
    if (args_.dry_run_latency &&
        (!args_.dry_run || args_.event_log.empty() || args_.check_only ||
         (variant_ != "allmimic" && variant_ != "supermimic" && variant_ != "uni_supermimic" && variant_ != "mjlab_combat") ||
         preview_mode_enabled())) {
      throw std::runtime_error("--dry-run-latency requires --dry-run --event-log and an AllMimic/SuperMimic/mjlab_combat spec without preview or --check-only");
    }
    event_log_.log("spec_loaded",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("spec", spec_path_.string()),
                    event_int("motor_count", motor_count_),
                    event_int("obs_dim", configured_obs_dim_),
                    event_int("action_dim", configured_action_dim_),
                    event_int("inference_hz", inference_hz_),
                    event_int("publish_hz", publish_hz_)});
    load_variant_params();
    if (variant_ != "sonic") {
      runtime_model_path_ = resolve_onnx_model(model_path_);
      if (!fs::exists(runtime_model_path_)) throw std::runtime_error("ONNX model not found: " + runtime_model_path_.string());
      onnx_.load(runtime_model_path_, onnx_threads_);
      onnx_.require_inputs(expected_input_names_);
      onnx_.require_outputs(expected_output_names_);
    }
    event_log_.log("model_loaded",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("model", variant_ == "sonic" ? sonic_decoder_path_.string() : runtime_model_path_.string())});
    post_load_variant();
    if (args_.check_only) {
      check_only();
      event_log_.log("check_only_ok",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_)});
      return;
    }

    if (args_.offline_dry_run_state) {
      if (!args_.dry_run) {
        throw std::runtime_error("--offline-dry-run-state requires --dry-run");
      }
      RobotState synthetic;
      synthetic.stamp = std::chrono::steady_clock::now();
      synthetic.connected = true;
      synthetic.mode_machine = mode_machine_;
      synthetic.quat = {1.f, 0.f, 0.f, 0.f};
      synthetic.q = default_q_;
      synthetic.q.resize(motor_count_, 0.f);
      synthetic.dq.assign(motor_count_, 0.f);
      synthetic.seq = 1;
      {
        std::lock_guard<std::mutex> lock(state_mtx_);
        state_ = std::move(synthetic);
      }
      connected_.store(true);
      event_log_.log("offline_state_seeded",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_int("motor_count", motor_count_)});
    } else {
      unitree::robot::ChannelFactory::Instance()->Init(args_.domain_id, net_);
      if (!args_.dry_run) {
        publisher_.reset(new unitree::robot::ChannelPublisher<LowCmd>(cmd_topic_));
        publisher_->InitChannel();
      }
      subscriber_.reset(new unitree::robot::ChannelSubscriber<LowState>(state_topic_));
      subscriber_->InitChannel(std::bind(&GenericPolicyPlayer::on_low_state, this, std::placeholders::_1), 1);
    }
    wait_for_state();
    (this->*runtime_module().control_loop)();
  }

void GenericPolicyPlayer::log_process_exit(bool ok, const std::string& reason) noexcept {
    try {
      event_log_.log("process_exit",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_bool("ok", ok),
                      event_string("reason", reason),
                      event_int("publish_count", active_publish_count_.load()),
                      event_number("max_publish_interval_ms",
                                   max_active_publish_interval_ms_.load())});
    } catch (...) {
    }
  }

void GenericPolicyPlayer::emergency_stop_after_error() noexcept {
    if (variant_ == "pbfm" && !ready_file_.empty()) {
      std::error_code error;
      fs::remove(ready_file_, error);
    }
    try {
      if (preview_mode_enabled()) {
        event_log_.log("candidate_fault_isolated",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("reason", "candidate_player_exception")});
        return;
      }
      if (!args_.dry_run && publisher_ && connected_.load()) {
        event_log_.log("safety_stop",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("reason", "exception_path_damping")});
        publish_damping_for_stop(true);
      }
    } catch (const std::exception& e) {
      std::cerr << player_log_prefix() << " emergency damping failed after error: " << e.what() << "\n";
    } catch (...) {
      std::cerr << player_log_prefix() << " emergency damping failed after unknown error\n";
    }
  }

void GenericPolicyPlayer::validate_player_scope() const {
    const PolicyContract* contract = contract_for_template(template_);
    if (!contract) {
      if (const PolicyContract* variant_contract = contract_for_variant(variant_)) {
        throw std::runtime_error("runtime.variant " + variant_ + " belongs to " +
                                 player_binary_name(variant_contract->player_kind) +
                                 "; spec runtime.template is unsupported for " +
                                 player_label() + ": " + template_);
      }
      throw std::runtime_error("unsupported runtime.template for " +
                               player_label() + ": " + template_ +
                               ". Supported templates: " +
                               format_supported_templates(player_kind()));
    }
    if (contract->player_kind != player_kind()) {
      throw std::runtime_error("runtime.template " + template_ +
                               " belongs to " + player_binary_name(contract->player_kind) +
                               "; " + player_label() + " supports only " +
                               format_supported_templates(player_kind()));
    }
    if (!variant_allowed_for_template(template_, variant_)) {
      throw std::runtime_error(std::string(contract->template_name) +
                               " supports variants: " +
                               format_supported_variants_for_template(template_));
    }
  }

const GenericPolicyPlayer::PolicyRuntimeModule& GenericPolicyPlayer::runtime_module() const {
    static const PolicyRuntimeModule kModules[] = {
        {"pbfm", GenericPlayerKind::kSensor,
         &GenericPolicyPlayer::load_spec_pbfm,
         &GenericPolicyPlayer::load_bfm_params,
         &GenericPolicyPlayer::post_load_pbfm,
         &GenericPolicyPlayer::schema_summary_bfm,
         &GenericPolicyPlayer::check_only_pbfm,
         &GenericPolicyPlayer::control_loop_single_bfm},
        {"bfm_zero", GenericPlayerKind::kPolicy,
         &GenericPolicyPlayer::load_spec_bfm,
         &GenericPolicyPlayer::load_bfm_params,
         &GenericPolicyPlayer::post_load_bfm,
         &GenericPolicyPlayer::schema_summary_bfm,
         &GenericPolicyPlayer::check_only_bfm,
         &GenericPolicyPlayer::control_loop_single_bfm},
        {"unitree_rl_mjlab", GenericPlayerKind::kPolicy,
         &GenericPolicyPlayer::load_spec_unitree,
         &GenericPolicyPlayer::load_unitree_params,
         &GenericPolicyPlayer::post_load_unitree,
         &GenericPolicyPlayer::schema_summary_unitree,
         &GenericPolicyPlayer::check_only_unitree,
         &GenericPolicyPlayer::control_loop_single_unitree},
        {"beyondmimic", GenericPlayerKind::kPolicy,
         &GenericPolicyPlayer::load_spec_beyondmimic,
         &GenericPolicyPlayer::load_beyond_params,
         &GenericPolicyPlayer::post_load_beyondmimic,
         &GenericPolicyPlayer::schema_summary_beyondmimic,
         &GenericPolicyPlayer::check_only_beyondmimic,
         &GenericPolicyPlayer::control_loop_beyondmimic},
        {"holosoma", GenericPlayerKind::kPolicy,
         &GenericPolicyPlayer::load_spec_holosoma,
         &GenericPolicyPlayer::load_holosoma_params,
         &GenericPolicyPlayer::post_load_holosoma,
         &GenericPolicyPlayer::schema_summary_holosoma,
         &GenericPolicyPlayer::check_only_holosoma,
         &GenericPolicyPlayer::control_loop_holosoma},
        {"protomotions", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_protomotions,
         &GenericPolicyPlayer::load_protomotions_params,
         &GenericPolicyPlayer::post_load_protomotions,
         &GenericPolicyPlayer::schema_summary_protomotions,
         &GenericPolicyPlayer::check_only_protomotions,
         &GenericPolicyPlayer::control_loop_protomotions},
        {"twist", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_twist,
         &GenericPolicyPlayer::load_twist_params,
         &GenericPolicyPlayer::post_load_twist,
         &GenericPolicyPlayer::schema_summary_twist,
         &GenericPolicyPlayer::check_only_twist,
         &GenericPolicyPlayer::control_loop_twist},
        {"humanoid_gpt", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_humanoid_gpt,
         &GenericPolicyPlayer::load_humanoid_gpt_params,
         &GenericPolicyPlayer::post_load_humanoid_gpt,
         &GenericPolicyPlayer::schema_summary_humanoid_gpt,
         &GenericPolicyPlayer::check_only_humanoid_gpt,
         &GenericPolicyPlayer::control_loop_humanoid_gpt},
        {"mjlab_combat", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_supermimic,
         &GenericPolicyPlayer::load_supermimic_params,
         &GenericPolicyPlayer::post_load_supermimic,
         &GenericPolicyPlayer::schema_summary_supermimic,
         &GenericPolicyPlayer::check_only_supermimic,
         &GenericPolicyPlayer::control_loop_supermimic},
        {"allmimic", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_supermimic,
         &GenericPolicyPlayer::load_supermimic_params,
         &GenericPolicyPlayer::post_load_supermimic,
         &GenericPolicyPlayer::schema_summary_supermimic,
         &GenericPolicyPlayer::check_only_supermimic,
         &GenericPolicyPlayer::control_loop_supermimic},
        {"uni_supermimic", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_supermimic,
         &GenericPolicyPlayer::load_supermimic_params,
         &GenericPolicyPlayer::post_load_supermimic,
         &GenericPolicyPlayer::schema_summary_supermimic,
         &GenericPolicyPlayer::check_only_supermimic,
         &GenericPolicyPlayer::control_loop_supermimic},
        {"supermimic", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_supermimic,
         &GenericPolicyPlayer::load_supermimic_params,
         &GenericPolicyPlayer::post_load_supermimic,
         &GenericPolicyPlayer::schema_summary_supermimic,
         &GenericPolicyPlayer::check_only_supermimic,
         &GenericPolicyPlayer::control_loop_supermimic},
        {"sonic", GenericPlayerKind::kMotionTracking,
         &GenericPolicyPlayer::load_spec_sonic,
         &GenericPolicyPlayer::load_sonic_params,
         &GenericPolicyPlayer::post_load_sonic,
         &GenericPolicyPlayer::schema_summary_sonic,
         &GenericPolicyPlayer::check_only_sonic,
         &GenericPolicyPlayer::control_loop_sonic},
    };
    for (const PolicyRuntimeModule& module : kModules) {
      if (variant_ == module.variant && player_kind() == module.player_kind) return module;
    }
    throw std::runtime_error("unsupported variant for " + player_label() + ": " + variant_);
  }

std::string GenericPolicyPlayer::join_names(const std::vector<std::string>& names) {
    std::ostringstream out;
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) out << ", ";
      out << names[i];
    }
    return out.str();
  }

std::string GenericPolicyPlayer::join_ints(const std::vector<int>& values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out << ", ";
      out << values[i];
    }
    return out.str();
  }

void GenericPolicyPlayer::require_name_list_equal(const std::string& field,
                                      const std::vector<std::string>& actual,
                                      const std::vector<const char*>& expected_raw) {
    if (expected_raw.empty()) return;
    const std::vector<std::string> expected = protocol_names(expected_raw);
    if (actual != expected) {
      throw std::runtime_error(field + " does not match protocol constraint: expected [" +
                               join_names(expected) + "], got [" + join_names(actual) + "]");
    }
  }

void GenericPolicyPlayer::bind_protocol_constraint() {
    protocol_ = protocol_constraint_for_variant(variant_);
    if (!protocol_) {
      throw std::runtime_error("missing PolicyProtocolConstraint for variant: " + variant_);
    }
  }

void GenericPolicyPlayer::validate_spec_against_protocol() const {
    if (!protocol_) throw std::runtime_error("internal error: protocol constraint not bound");
    if (protocol_->motor_count > 0 && motor_count_ != protocol_->motor_count) {
      throw std::runtime_error("robot.motor_count protocol mismatch for " + variant_ +
                               ": expected " + std::to_string(protocol_->motor_count) +
                               ", got " + std::to_string(motor_count_));
    }
    if (protocol_->obs_dim > 0 && configured_obs_dim_ != protocol_->obs_dim) {
      throw std::runtime_error("observation.obs_dim protocol mismatch for " + variant_ +
                               ": expected " + std::to_string(protocol_->obs_dim) +
                               ", got " + std::to_string(configured_obs_dim_));
    }
    if (protocol_->action_dim > 0 && configured_action_dim_ != protocol_->action_dim) {
      throw std::runtime_error("action.action_dim protocol mismatch for " + variant_ +
                               ": expected " + std::to_string(protocol_->action_dim) +
                               ", got " + std::to_string(configured_action_dim_));
    }
    require_name_list_equal("model.input_names", expected_input_names_, protocol_->model_input_names);
    require_name_list_equal("model.output_names", expected_output_names_, protocol_->model_output_names);
  }

void GenericPolicyPlayer::load_spec() {
    spec_path_ = fs::absolute(fs::path(args_.spec)).lexically_normal();
    if (!fs::exists(spec_path_)) throw std::runtime_error("spec not found: " + spec_path_.string());
    const YAML::Node y = YAML::LoadFile(spec_path_.string());
    name_ = scalar_or<std::string>(y, "name", spec_path_.stem().string());
    const YAML::Node runtime = y["runtime"];
    const YAML::Node model = y["model"];
    const YAML::Node robot = y["robot"];
    const YAML::Node obs = y["observation"];
    const YAML::Node action = y["action"];
    const YAML::Node mimic = y["mimic_time"];
    const YAML::Node reference = y["reference"];
    const YAML::Node motion_command = y["motion_command"];
    const YAML::Node handoff = y["handoff"];
    const YAML::Node safety = y["safety"] ? y["safety"] : runtime["safety"];

    template_ = lower(scalar_or<std::string>(runtime, "template", ""));
    variant_ = lower(scalar_or<std::string>(runtime, "variant", scalar_or<std::string>(obs, "builder", "")));
    if (template_.empty()) throw std::runtime_error("spec runtime.template is required");
    if (variant_.empty()) throw std::runtime_error("spec runtime.variant or observation.builder is required");
    validate_player_scope();
    bind_protocol_constraint();

    motor_count_ = yamlc::required_scalar<int>(y, "robot.motor_count");
    if (motor_count_ <= 0) throw std::runtime_error("robot.motor_count must be positive");
    configured_obs_dim_ = yamlc::required_scalar<int>(y, "observation.obs_dim");
    if (configured_obs_dim_ <= 0) throw std::runtime_error("observation.obs_dim must be positive");
    configured_action_dim_ = yamlc::required_scalar<int>(y, "action.action_dim");
    if (configured_action_dim_ <= 0) throw std::runtime_error("action.action_dim must be positive");
    num_actions_ = configured_action_dim_;
    num_obs_ = configured_obs_dim_;

    if (variant_ == "sonic") {
      const YAML::Node sessions = yamlc::required_node(y, "model.sessions");
      sonic_encoder_path_ = resolve_relative(spec_path_, yamlc::required_scalar<std::string>(sessions, "encoder.path"));
      sonic_decoder_path_ = resolve_relative(spec_path_, yamlc::required_scalar<std::string>(sessions, "decoder.path"));
      sonic_planner_path_ = resolve_relative(spec_path_, yamlc::required_scalar<std::string>(sessions, "planner.path"));
      model_path_ = sonic_decoder_path_;
    } else {
      std::string raw_model = yamlc::required_scalar<std::string>(y, "model.path");
      if (!args_.model_override.empty()) raw_model = args_.model_override;
      model_path_ = resolve_relative(spec_path_, raw_model);
    }

    std::string raw_params = yamlc::required_scalar<std::string>(y, "robot.params");
    if (!args_.params_override.empty()) raw_params = args_.params_override;
    params_path_ = resolve_relative(spec_path_, raw_params);

    std::string raw_trajectory = scalar_or<std::string>(reference, "path", scalar_or<std::string>(reference, "trajectory", ""));
    if (!args_.trajectory_override.empty()) raw_trajectory = args_.trajectory_override;
    if (!raw_trajectory.empty()) trajectory_path_ = resolve_relative(spec_path_, raw_trajectory);

    latent_file_ = scalar_or<std::string>(obs, "latent_file", scalar_or<std::string>(model, "latent_file", ""));
    if (!args_.latent_file_override.empty()) latent_file_ = args_.latent_file_override;

    net_ = args_.net.empty() ? scalar_or<std::string>(runtime, "net", "enP8p1s0") : args_.net;
    cmd_topic_ = args_.cmd_topic.empty() ? yamlc::required_scalar<std::string>(y, "handoff.cmd_topic") : args_.cmd_topic;
    state_topic_ = args_.state_topic.empty() ? yamlc::required_scalar<std::string>(y, "handoff.state_topic") : args_.state_topic;
    inference_hz_ = args_.inference_hz > 0 ? args_.inference_hz : yamlc::required_scalar<int>(y, "runtime.inference_hz");
    publish_hz_ = args_.publish_hz > 0 ? args_.publish_hz : yamlc::required_scalar<int>(y, "runtime.publish_hz");
    sim_lowstate_decimation_ = args_.sim_lowstate_decimation;
    if (sim_lowstate_decimation_ < 0) {
      throw std::runtime_error("--sim-lowstate-decimation must be non-negative");
    }
    onnx_threads_ = args_.onnx_threads > 0 ? args_.onnx_threads : yamlc::required_scalar<int>(y, "runtime.onnx_threads");
    duration_sec_ = args_.duration_sec >= 0.0 ? args_.duration_sec : scalar_or<double>(runtime, "duration_sec", 0.0);
    if (args_.dry_run && duration_sec_ <= 0.0) duration_sec_ = scalar_or<double>(runtime, "dry_run_duration_sec", 1.0);
    target_interp_sec_ = scalar_or<double>(runtime, "target_interp_sec", scalar_or<double>(action, "target_interp_sec", 0.02));
    activation_blend_sec_ = std::isfinite(args_.activation_blend_sec)
        ? args_.activation_blend_sec
        : scalar_or<double>(handoff, "activation_blend_sec", 0.10);
    activation_blend_.set_duration(activation_blend_sec_);
    startup_interp_sec_ = scalar_or<double>(runtime, "startup_interp_sec", scalar_or<double>(action, "startup_interp_sec", 0.0));
    mode_machine_ = yamlc::required_scalar<int>(y, "handoff.mode_machine");
    ready_file_ = args_.ready_file.empty() ? scalar_or<std::string>(handoff, "ready_file", "") : args_.ready_file;
    start_file_ = args_.start_file.empty() ? scalar_or<std::string>(handoff, "start_file", "") : args_.start_file;
    handoff_complete_file_ = args_.handoff_complete_file;
    handoff_release_file_ = args_.handoff_release_file;
    if (handoff_complete_file_.empty() != handoff_release_file_.empty()) {
      throw std::runtime_error(
          "--handoff-complete-file and --handoff-release-file must be provided together");
    }
    preview_file_ = args_.preview_file;
    commit_file_ = args_.commit_file;
    discard_file_ = args_.discard_file;
    preview_timeout_sec_ = args_.preview_timeout_sec;
    if (preview_mode_enabled()) {
      if (preview_file_.empty() || commit_file_.empty()) {
        throw std::runtime_error(
            "transactional preview requires --preview-file and --commit-file");
      }
      if (!std::isfinite(preview_timeout_sec_) || preview_timeout_sec_ <= 0.0) {
        throw std::runtime_error("--preview-timeout-sec must be positive");
      }
      const PolicyVariantContract* preview_contract = variant_contract(variant_);
      if (!preview_contract ||
          preview_contract->preview_capability != PreviewCapability::kTransactional) {
        throw std::runtime_error(
            "runtime.variant does not declare transactional preview capability: " +
            variant_);
      }
      if (cmd_topic_ == "rt/user_lowcmd" || cmd_topic_ == "rt/lowcmd") {
        throw std::runtime_error(
            "transactional preview requires a private --candidate-topic; refusing the real LowCmd topic");
      }
      if (fs::exists(commit_file_) ||
          (!discard_file_.empty() && fs::exists(discard_file_))) {
        throw std::runtime_error(
            "stale preview lifecycle input exists; remove commit/discard files before launch");
      }
      std::error_code remove_error;
      fs::remove(preview_file_, remove_error);
      event_log_.log("policy_entry_contract",
                     {event_int("contract_schema_version", 1),
                      event_string("contract_policy", variant_),
                      event_string("preview_capability",
                                   preview_capability_name(preview_contract->preview_capability)),
                      event_bool("freezes_external_input",
                                 preview_contract->freezes_external_input),
                      event_number("activation_horizon_sec", 2.0),
                      event_string("candidate_topic", cmd_topic_)});
    }
    damping_on_stop_ = args_.damping_on_stop && yamlc::required_scalar<bool>(y, "handoff.damping_on_stop");
    stats_interval_ = scalar_or<int>(runtime, "stats_interval", 100);
    command_safety_config_.profile = lower(scalar_or<std::string>(safety, "profile", "hardware"));
    if (command_safety_config_.profile != "hardware" &&
        command_safety_config_.profile != "simulation_parity") {
      throw std::runtime_error("safety.profile must be hardware or simulation_parity");
    }
    const bool simulation_parity = command_safety_config_.profile == "simulation_parity";
    command_safety_config_.enabled = scalar_or<bool>(safety, "enabled", true);
    command_safety_config_.finite_check = scalar_or<bool>(safety, "finite_check", true);
    command_safety_config_.joint_limit_enabled =
        scalar_or<bool>(safety, "joint_limit_enabled", !simulation_parity);
    command_safety_config_.preserve_pd_torque = scalar_or<bool>(safety, "preserve_pd_torque", false);
    if (command_safety_config_.preserve_pd_torque && variant_ != "allmimic" && variant_ != "supermimic") {
      throw std::runtime_error("generic safety.preserve_pd_torque requires allmimic or supermimic");
    }
    command_safety_config_.lowstate_timeout_enabled = scalar_or<bool>(safety, "lowstate_timeout_enabled", true);
    command_safety_config_.lowstate_timeout_sec = args_.lowstate_timeout_sec > 0.0
        ? args_.lowstate_timeout_sec : scalar_or<double>(safety, "lowstate_timeout_sec", 0.2);
    command_safety_config_.kp_max = scalar_or<float>(safety, "kp_max", 300.f);
    command_safety_config_.kd_max = scalar_or<float>(safety, "kd_max", 20.f);
    command_safety_config_.use_current_q_for_missing = scalar_or<bool>(safety, "use_current_q_for_missing", true);
    command_safety_config_.fallback_q = scalar_or<float>(safety, "fallback_q", 0.f);
    const float max_step = scalar_or<float>(safety, "max_position_step_rad",
                                            scalar_or<float>(safety, "max_step_rad", -1.f));
    command_safety_config_.max_position_step_enabled =
        scalar_or<bool>(safety, "max_position_step_enabled", max_step > 0.f);
    if (max_step > 0.f) command_safety_config_.max_position_step_rad = max_step;
    const auto require_finite = [](double value, const char* name) {
      if (!std::isfinite(value)) throw std::runtime_error(std::string("safety.") + name + " must be finite");
    };
    require_finite(command_safety_config_.lowstate_timeout_sec, "lowstate_timeout_sec");
    require_finite(command_safety_config_.kp_max, "kp_max");
    require_finite(command_safety_config_.kd_max, "kd_max");
    require_finite(command_safety_config_.fallback_q, "fallback_q");
    require_finite(command_safety_config_.max_position_step_rad, "max_position_step_rad");
    if (command_safety_config_.lowstate_timeout_sec <= 0.0 ||
        command_safety_config_.lowstate_timeout_sec > 10.0 ||
        command_safety_config_.kp_max < 0.f || command_safety_config_.kp_max > 1000.f ||
        command_safety_config_.kd_max < 0.f || command_safety_config_.kd_max > 100.f ||
        command_safety_config_.max_position_step_rad < 0.f ||
        command_safety_config_.max_position_step_rad > 10.f) {
        throw std::runtime_error("safety numeric value is outside deployment bounds");
    }
    if (simulation_parity &&
        (command_safety_config_.joint_limit_enabled || command_safety_config_.max_position_step_enabled)) {
      throw std::runtime_error(
          "simulation_parity safety forbids joint-limit and max-position-step target clamps");
    }
    safety_filter_.set_config(command_safety_config_);
    safety_limits_.lower.clear();
    safety_limits_.upper.clear();

    if (variant_ == "sonic") {
      expected_input_names_.clear();
      expected_output_names_.clear();
    } else {
      expected_input_names_ = yamlc::required_vector<std::string>(y, "model.input_names");
      expected_output_names_ = yamlc::required_vector<std::string>(y, "model.output_names");
      if (expected_input_names_.empty()) throw std::runtime_error("model.input_names must not be empty");
      if (expected_output_names_.empty()) throw std::runtime_error("model.output_names must not be empty");
    }
    validate_spec_against_protocol();
    obs_input_name_ = scalar_or<std::string>(model, "obs_input", expected_input_names_.empty() ? "" : expected_input_names_.front());
    time_input_name_ = scalar_or<std::string>(mimic, "time_input_name", scalar_or<std::string>(model, "time_input", "time_step"));
    action_output_name_ = scalar_or<std::string>(model, "action_output", expected_output_names_.empty() ? "" : expected_output_names_.front());
    twist_action_scale_ = std::isfinite(args_.action_scale_override)
                              ? args_.action_scale_override
                              : scalar_or<float>(action, "scale", scalar_or<float>(action, "action_scale", std::numeric_limits<float>::quiet_NaN()));

    const int spec_max_frames = scalar_or<int>(runtime, "max_frames", scalar_or<int>(mimic, "max_frames", 0));
    max_frames_ = args_.max_frames >= 0 ? args_.max_frames : spec_max_frames;
    (this->*runtime_module().load_spec_fields)(y);
  }

void GenericPolicyPlayer::load_variant_params() {
    (this->*runtime_module().load_params)();
    yamlc::require_size("robot.joint_pos_lower_limit", motor_count_, lower_limits_.size());
    yamlc::require_size("robot.joint_pos_upper_limit", motor_count_, upper_limits_.size());
    if (!joint_index_.empty()) {
      validate_joint_index();
      safety_limits_.lower = model_to_motor(lower_limits_);
      safety_limits_.upper = model_to_motor(upper_limits_);
    } else {
      safety_limits_.lower = lower_limits_;
      safety_limits_.upper = upper_limits_;
    }
    for (size_t i = 0; i < motor_count_; ++i) {
      if (!std::isfinite(safety_limits_.lower[i]) || !std::isfinite(safety_limits_.upper[i]) ||
          safety_limits_.lower[i] > safety_limits_.upper[i]) {
        throw std::runtime_error("robot joint limits must be finite and lower <= upper");
      }
    }
}

void GenericPolicyPlayer::post_load_variant() {
    (this->*runtime_module().post_load)();
  }

void GenericPolicyPlayer::validate_joint_index() const {
    std::vector<int> seen(motor_count_, 0);
    for (int i = 0; i < motor_count_; ++i) {
      const int motor_i = joint_index_[i];
      if (motor_i < 0 || motor_i >= motor_count_) throw std::runtime_error("joint_index contains out-of-range motor index");
      if (++seen[motor_i] > 1) throw std::runtime_error("joint_index is not a permutation");
    }
  }

void GenericPolicyPlayer::schema_summary_common() const {
    if (!protocol_) return;
    std::cout << "  protocol=" << format_protocol_constraint_summary(*protocol_) << "\n";
  }

void GenericPolicyPlayer::check_only() {
    std::cout << player_log_prefix() << " check-only\n"
              << "  name=" << name_ << " template=" << template_ << " variant=" << variant_ << "\n"
              << "  contract_player=" << player_label()
              << " motor_count=" << motor_count_
              << " obs_dim=" << configured_obs_dim_
              << " action_dim=" << configured_action_dim_ << "\n"
              << "  spec=" << spec_path_.string() << "\n"
              << "  model=" << (variant_ == "sonic" ? sonic_decoder_path_.string() : runtime_model_path_.string()) << "\n"
              << "  params=" << params_path_.string() << "\n"
              << "  inputs=[";
    print_names(variant_ == "sonic" ? sonic_decoder_.input_names() : onnx_.input_names());
    std::cout << "] outputs=[";
    print_names(variant_ == "sonic" ? sonic_decoder_.output_names() : onnx_.output_names());
    std::cout << "]\n";
    (this->*runtime_module().schema_summary)();
    (this->*runtime_module().check_only)();
    std::cout << player_log_prefix() << " check-only OK\n";
  }

void GenericPolicyPlayer::print_names(const std::vector<std::string>& names) {
    for (size_t i = 0; i < names.size(); ++i) {
      std::cout << (i ? "," : "") << names[i];
    }
  }

int GenericPolicyPlayer::advance_deadline(std::chrono::steady_clock::time_point& next,
                              std::chrono::steady_clock::duration dt,
                              std::chrono::steady_clock::time_point now) {
    int missed = 0;
    do {
      next += dt;
      if (next <= now) ++missed;
    } while (next <= now);
    return missed;
  }

void GenericPolicyPlayer::on_low_state(const void* message) {
    const auto received_at = std::chrono::steady_clock::now();
    const auto low = *(const LowState*)message;
    if (low.motor_state().size() < motor_count_) return;
    const auto quat = low.imu_state().quaternion();
    const auto gyro = low.imu_state().gyroscope();
    const auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(quat.begin(), quat.end(), finite) ||
        !std::all_of(gyro.begin(), gyro.end(), finite)) return;
    RobotState s;
    s.stamp = received_at;
    s.connected = true;
    s.mode_machine = static_cast<int>(low.mode_machine());
    s.source_tick_ms = low.tick();
    s.quat = low.imu_state().quaternion();
    s.gyro = low.imu_state().gyroscope();
    float quat_norm_sq = 0.f;
    bool imu_valid = true;
    for (float value : s.quat) {
      imu_valid = imu_valid && std::isfinite(value);
      quat_norm_sq += value * value;
    }
    for (float value : s.gyro) imu_valid = imu_valid && std::isfinite(value);
    if (!imu_valid || quat_norm_sq < 1e-8f) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        std::cerr << player_log_prefix()
                  << " rejected LowState with invalid IMU quaternion/gyroscope\n";
      }
      connected_.store(false);
      return;
    }
    s.q.assign(motor_count_, 0.f);
    s.dq.assign(motor_count_, 0.f);
    for (size_t i = 0; i < motor_count_; ++i) {
      s.q[i] = low.motor_state()[i].q();
      s.dq[i] = low.motor_state()[i].dq();
      if (!finite(s.q[i]) || !finite(s.dq[i])) return;
    }
    const auto& remote = low.wireless_remote();
    for (size_t i = 0; i < s.wireless_remote.size() && i < remote.size(); ++i) {
      s.wireless_remote[i] = static_cast<uint8_t>(remote[i]);
    }
    {
      std::lock_guard<std::mutex> lock(state_mtx_);
      s.seq = state_.seq + 1;
      state_ = std::move(s);
    }
    connected_.store(true);
  }

GenericPolicyPlayer::RobotState GenericPolicyPlayer::snapshot() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return state_;
  }

CommandSafetyState GenericPolicyPlayer::command_safety_state(const GenericPolicyPlayer::RobotState& s) const {
    CommandSafetyState out;
    out.connected = s.connected;
    out.stamp = s.stamp;
    out.seq = s.seq;
    out.q = s.q;
    out.dq = s.dq;
    return out;
  }

void GenericPolicyPlayer::wait_for_state() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!connected_.load() && std::chrono::steady_clock::now() < deadline && !g_stop_requested.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!connected_.load()) throw std::runtime_error("timeout waiting for lowstate");
    std::cout << player_log_prefix() << " lowstate connected variant=" << variant_
              << " dry_run=" << (args_.dry_run ? "true" : "false");
    std::cout << "\n";
  }

void GenericPolicyPlayer::touch_file(const std::string& path) {
    if (path.empty()) return;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream out(path);
    if (!out.good()) throw std::runtime_error("failed to touch file: " + path);
    out << "ready\n";
    event_log_.log("ready_file_touched",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("path", path)});
  }

bool GenericPolicyPlayer::preview_mode_enabled() const {
    return !preview_file_.empty() || !commit_file_.empty() || !discard_file_.empty();
  }

GenericPolicyPlayer::PreviewDecision GenericPolicyPlayer::wait_preview_decision(
    const std::vector<float>& q,
    const std::vector<float>& kp,
    const std::vector<float>& kd,
    int mode_machine,
    int publish_hz,
    int warmup_steps) {
    if (!preview_mode_enabled()) return PreviewDecision::kDisabled;

    const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, publish_hz)));
    const auto start = std::chrono::steady_clock::now();
    auto next_publish = start;
    publish(q, kp, kd, mode_machine, false, CommandPhase::kPreview);
    {
      fs::path preview_path(preview_file_);
      if (preview_path.has_parent_path()) fs::create_directories(preview_path.parent_path());
      std::ofstream preview_stream(preview_file_, std::ios::out | std::ios::trunc);
      if (!preview_stream.good()) {
        throw std::runtime_error("failed to write preview file: " + preview_file_);
      }
      preview_stream << "preview_ready\n";
    }
    event_log_.log("preview_ready",
                   {event_string("player", player_label()),
                    event_string("template", template_),
                    event_string("variant", variant_),
                    event_string("candidate_topic", cmd_topic_),
                    event_string("preview_file", preview_file_),
                    event_int("warmup_steps", warmup_steps)});

    while (!g_stop_requested.load()) {
      if (!discard_file_.empty() && fs::exists(discard_file_)) {
        event_log_.log("preview_discarded",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("discard_file", discard_file_)});
        return PreviewDecision::kDiscard;
      }
      if (fs::exists(commit_file_)) {
        event_log_.log("preview_committed",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("commit_file", commit_file_)});
        return PreviewDecision::kCommit;
      }
      const auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - start).count() >= preview_timeout_sec_) {
        event_log_.log("preview_timeout",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_number("timeout_sec", preview_timeout_sec_)});
        return PreviewDecision::kTimeout;
      }
      if (now >= next_publish) {
        publish(q, kp, kd, mode_machine, false, CommandPhase::kPreview);
        next_publish = now + period;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return PreviewDecision::kStopped;
  }

void GenericPolicyPlayer::publish(const std::vector<float>& q,
               const std::vector<float>& kp,
               const std::vector<float>& kd,
               int mode_machine,
               bool allow_stale_state,
               CommandPhase phase) {
    if (!args_.dry_run_latency && (args_.dry_run || !publisher_)) return;
    if (phase == CommandPhase::kHold && args_.experimental_no_prestart_hold) {
      if (!prestart_hold_suppressed_logged_) {
        prestart_hold_suppressed_logged_ = true;
        event_log_.log("prestart_hold_suppressed",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("reason", "experimental_handoff_ablation")});
      }
      return;
    }
    const auto safety_state = command_safety_state(snapshot());
    const bool preview_commit_bypass =
        phase == CommandPhase::kActive && !preview_candidate_q_.empty();
    JointCommand command{q, kp, kd};
    ActivationBlendResult blend_result;
    if (phase == CommandPhase::kActive) {
      blend_result = activation_blend_.apply(command, ActivationBlend::Clock::now(),
                                             preview_commit_bypass);
      command = std::move(blend_result.command);
      if (blend_result.started) {
        event_log_.log("activation_blend_started",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_number("duration_sec", activation_blend_sec_),
                        event_number("max_q_delta", blend_result.max_q_delta),
                        event_number("max_kp_delta", blend_result.max_kp_delta),
                        event_number("max_kd_delta", blend_result.max_kd_delta)});
      }
      if (preview_commit_bypass && !activation_blend_bypass_logged_) {
        activation_blend_bypass_logged_ = true;
        event_log_.log("activation_blend_bypassed",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_string("reason", "preview_commit")});
      }
    }
    FilteredCommand filtered;
    try {
      filtered = safety_filter_.filter(command.q, command.kp, command.kd,
                                       safety_state, safety_limits_,
                                       motor_count_, allow_stale_state);
    } catch (const std::exception& e) {
      event_log_.log("safety_stop",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("reason", e.what()),
                      event_int("state_seq", static_cast<long long>(safety_state.seq))});
      throw;
    }
    LowCmd cmd;
    cmd.mode_pr() = 0;
    cmd.mode_machine() = static_cast<uint8_t>(mode_machine > 0 ? mode_machine : mode_machine_);
    for (int i = 0; i < motor_count_ && i < static_cast<int>(cmd.motor_cmd().size()); ++i) {
      cmd.motor_cmd()[i].mode() = 1;
      cmd.motor_cmd()[i].q() = filtered.q[i];
      cmd.motor_cmd()[i].dq() = 0.f;
      cmd.motor_cmd()[i].tau() = filtered.tau[i];
      cmd.motor_cmd()[i].kp() = filtered.kp[i];
      cmd.motor_cmd()[i].kd() = filtered.kd[i];
    }
    if (beyond_sim_lockstep_sequence_) {
      constexpr uint32_t kActionGoLockstepMagic = 0x41474c53u;
      cmd.reserve()[0] = static_cast<uint32_t>(policy_counter_);
      cmd.reserve()[1] = phase == CommandPhase::kActive
          ? kActionGoLockstepMagic : 0u;
    }
    if (supermimic_sim_lockstep_sequence_) {
      cmd.reserve()[0] = supermimic_command_sequence_.load();
      cmd.reserve()[1] = phase == CommandPhase::kActive ? 0x41474c53u : 0u;
    }
    cmd.crc() = crc32_core((uint32_t*)&cmd, (sizeof(cmd) >> 2) - 1);
    if (args_.dry_run) {
      dry_run_command_crc_ = cmd.crc();
      last_published_q_ = filtered.q;
      last_published_kp_ = filtered.kp;
      last_published_kd_ = filtered.kd;
      if (phase == CommandPhase::kHold) {
        activation_blend_.observe_hold({filtered.q, filtered.kp, filtered.kd});
      }
      return;
    }
    publisher_->Write(cmd);
    if (phase == CommandPhase::kActive) {
      const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count();
      const int64_t previous_ns = last_active_publish_ns_.exchange(now_ns);
      active_publish_count_.fetch_add(1);
      if (previous_ns > 0) {
        const double interval_ms = static_cast<double>(now_ns - previous_ns) / 1e6;
        double observed = max_active_publish_interval_ms_.load();
        while (interval_ms > observed &&
               !max_active_publish_interval_ms_.compare_exchange_weak(observed, interval_ms)) {
        }
      }
    }
    last_published_q_ = filtered.q;
    last_published_kp_ = filtered.kp;
    last_published_kd_ = filtered.kd;
    if (phase == CommandPhase::kHold) {
      activation_blend_.observe_hold({filtered.q, filtered.kp, filtered.kd});
    }
    if (phase == CommandPhase::kActive && blend_result.completed &&
        !blend_result.bypassed && activation_blend_sec_ > 0.0) {
      event_log_.log("activation_blend_completed",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_number("duration_sec", activation_blend_sec_)});
    }
    const char* phase_name = phase == CommandPhase::kHold
        ? "hold"
        : (phase == CommandPhase::kPreview
               ? "preview"
               : (phase == CommandPhase::kDamping ? "damping" : "active"));
    if (!first_command_logged_) {
      first_command_logged_ = true;
      event_log_.log("first_command_published",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("command_phase", phase_name),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", mode_machine)});
    }
    if (phase == CommandPhase::kHold && !first_hold_logged_) {
      first_hold_logged_ = true;
      event_log_.log("first_hold_published",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", mode_machine)});
    }
    if (phase == CommandPhase::kPreview && !first_preview_candidate_logged_) {
      first_preview_candidate_logged_ = true;
      preview_candidate_q_ = q;
      preview_candidate_kp_ = kp;
      preview_candidate_kd_ = kd;
      event_log_.log("first_preview_candidate",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", mode_machine)});
    }
    if (phase == CommandPhase::kActive && !first_active_command_logged_) {
      first_active_command_logged_ = true;
      event_log_.log("first_active_command",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_int("state_seq", static_cast<long long>(safety_state.seq)),
                      event_int("mode_machine", mode_machine)});
      if (!preview_candidate_q_.empty() && !preview_active_consistency_logged_) {
        preview_active_consistency_logged_ = true;
        const auto max_abs_diff = [](const std::vector<float>& candidate,
                                     const std::vector<float>& active) {
          double result = 0.0;
          const size_t count = std::min(candidate.size(), active.size());
          for (size_t index = 0; index < count; ++index) {
            result = std::max(
                result,
                std::abs(static_cast<double>(candidate[index]) - active[index]));
          }
          return result;
        };
        const double max_q_diff = max_abs_diff(preview_candidate_q_, q);
        const double max_kp_diff = max_abs_diff(preview_candidate_kp_, kp);
        const double max_kd_diff = max_abs_diff(preview_candidate_kd_, kd);
        const bool dimensions_match = preview_candidate_q_.size() == q.size() &&
            preview_candidate_kp_.size() == kp.size() &&
            preview_candidate_kd_.size() == kd.size();
        event_log_.log("preview_active_first_consistency",
                       {event_string("player", player_label()),
                        event_string("template", template_),
                        event_string("variant", variant_),
                        event_bool("match", dimensions_match && max_q_diff <= 1e-6 &&
                                                max_kp_diff <= 1e-6 && max_kd_diff <= 1e-6),
                        event_number("max_abs_diff", std::max({max_q_diff, max_kp_diff, max_kd_diff})),
                        event_number("max_q_diff", max_q_diff),
                        event_number("max_kp_diff", max_kp_diff),
                        event_number("max_kd_diff", max_kd_diff)});
      }
    }
  }

void GenericPolicyPlayer::publish_damping_for_stop(bool allow_stale_state ) {
    if (args_.dry_run || !publisher_) return;
    const auto s = snapshot();
    std::vector<float> q = s.q;
    q.resize(motor_count_, 0.f);
    std::vector<float> kp(motor_count_, 0.f), kd(motor_count_, 3.f);
    for (int i = 0; i < 20; ++i) {
      publish(q, kp, kd, s.mode_machine, allow_stale_state, CommandPhase::kDamping);
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

void GenericPolicyPlayer::hold_current_pose(const GenericPolicyPlayer::RobotState& s, std::vector<float>& q, std::vector<float>& kp, std::vector<float>& kd,
                         float hold_kp , float hold_kd ) const {
    q.assign(motor_count_, 0.f);
    kp.assign(motor_count_, hold_kp);
    kd.assign(motor_count_, hold_kd);
    for (int i = 0; i < motor_count_; ++i) q[i] = i < static_cast<int>(s.q.size()) ? s.q[i] : 0.f;
  }

void GenericPolicyPlayer::wait_postrun_handoff(int publish_hz) {
    if (g_stop_requested.load() || handoff_complete_file_.empty()) return;
    const auto target_q = last_published_q_;
    const auto target_kp = last_published_kp_;
    const auto target_kd = last_published_kd_;
    if (!args_.dry_run && target_q.size() != static_cast<size_t>(motor_count_))
      throw std::runtime_error("postrun handoff requires a published target");
    const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, publish_hz)));
    event_log_.log("handoff_hold_started",
                   {event_string("player", player_label()), event_string("template", template_),
                    event_string("variant", variant_), event_string("complete_file", handoff_complete_file_),
                    event_string("release_file", handoff_release_file_)});
    publish(target_q, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kHold);
    touch_file(handoff_complete_file_);
    auto next = std::chrono::steady_clock::now();
    while (!g_stop_requested.load() && !fs::exists(handoff_release_file_)) {
      publish(target_q, target_kp, target_kd, snapshot().mode_machine, false, CommandPhase::kHold);
      advance_deadline(next, interval, std::chrono::steady_clock::now());
      std::this_thread::sleep_until(next);
    }
    event_log_.log("handoff_hold_stopped",
                   {event_string("player", player_label()), event_string("template", template_),
                    event_string("variant", variant_), event_bool("released", fs::exists(handoff_release_file_)),
                    event_string("reason", g_stop_requested.load() ? "signal" : "release_file")});
  }

void GenericPolicyPlayer::wait_start_file(std::vector<float>& hold_q,
                                          const std::vector<float>& hold_kp,
                                          const std::vector<float>& hold_kd,
                                          int publish_hz,
                                          const std::function<void()>& standby_health_check,
                                          bool follow_current_pose) {
    if (start_file_.empty()) return;
    std::cout << player_log_prefix() << " holding until start_file=" << start_file_ << "\n";
    const auto dt = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, publish_hz)));
    auto next = std::chrono::steady_clock::now();
    while (!g_stop_requested.load() && !fs::exists(start_file_)) {
      if (standby_health_check) standby_health_check();
      const auto s = snapshot();
      if (follow_current_pose) {
        hold_q = s.q;
        hold_q.resize(motor_count_, 0.f);
      }
      publish(hold_q, hold_kp, hold_kd, s.mode_machine, false, CommandPhase::kHold);
      next += dt;
      if (next < std::chrono::steady_clock::now()) next = std::chrono::steady_clock::now();
      std::this_thread::sleep_until(next);
    }
    if (!start_file_seen_logged_ && fs::exists(start_file_)) {
      if (standby_health_check) standby_health_check();
      if (follow_current_pose) {
        hold_q = snapshot().q;
        hold_q.resize(motor_count_, 0.f);
      }
      start_file_seen_logged_ = true;
      event_log_.log("start_file_seen",
                     {event_string("player", player_label()),
                      event_string("template", template_),
                      event_string("variant", variant_),
                      event_string("path", start_file_)});
    }
  }

}  // namespace action_go::runtime::generic_policy_runtime

namespace action_go::runtime {

int run_generic_policy_runtime(int argc, char** argv, GenericPolicyRuntimeScope scope) {
  generic_policy_runtime::set_runtime_scope(scope);
  generic_policy_runtime::g_stop_requested.store(false);
  std::signal(SIGINT, generic_policy_runtime::signal_handler);
  std::signal(SIGTERM, generic_policy_runtime::signal_handler);
  try {
    generic_policy_runtime::Args args = generic_policy_runtime::parse_args(argc, argv);
    generic_policy_runtime::GenericPolicyPlayer player(std::move(args));
    try {
      player.run();
      player.log_process_exit(true, "ok");
    } catch (...) {
      try {
        throw;
      } catch (const std::exception& e) {
        player.log_process_exit(false, e.what());
      } catch (...) {
        player.log_process_exit(false, "unknown_exception");
      }
      player.emergency_stop_after_error();
      throw;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << generic_policy_runtime::player_log_prefix() << " ERROR: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace action_go::runtime
