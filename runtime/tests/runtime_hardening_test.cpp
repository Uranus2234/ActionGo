#include "action_go/runtime/activation_blend.h"
#include "action_go/runtime/async_command_publisher.h"
#include "action_go/runtime/command_safety_filter.h"
#include "action_go/runtime/input_freshness.h"
#include "action_go/runtime/network_command.h"
#include "action_go/runtime/npy_reader.h"
#include "action_go/runtime/onnx_runner.h"
#include "action_go/runtime/policy_contract.h"
#include "action_go/runtime/policy_protocol.h"
#include "action_go/runtime/prearm_command.h"
#include "action_go/runtime/reference_endpoint.h"
#include "action_go/runtime/sonic_planner.h"
#include "action_go/runtime/sonic_policy.h"
#include "action_go/runtime/supermimic_policy.h"
#include "action_go/runtime/allmimic_policy.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <sstream>

#include <nlohmann/json.hpp>
#include <openssl/evp.h>
#include <openssl/hmac.h>

namespace {

void expect_true(const std::string& name, bool ok);
void expect_near(const std::string& name, float actual, float expected, float eps = 1e-6f);
void expect_throw(const std::string& name, const std::function<void()>& fn);

void test_activation_blend() {
  using action_go::runtime::ActivationBlend;
  using action_go::runtime::JointCommand;
  const JointCommand hold{{0.f, 2.f}, {20.f, 30.f}, {1.f, 2.f}};
  JointCommand active{{2.f, 4.f}, {40.f, 50.f}, {3.f, 4.f}};
  ActivationBlend blend(0.10);
  blend.observe_hold(hold);
  const auto t0 = ActivationBlend::Clock::time_point(std::chrono::seconds(10));
  auto out = blend.apply(active, t0);
  expect_true("activation blend starts on first active command", out.started);
  expect_near("activation blend q starts at hold", out.command.q[0], 0.f);
  expect_near("activation blend kp starts at hold", out.command.kp[0], 20.f);
  expect_near("activation blend kd starts at hold", out.command.kd[0], 1.f);

  active = {{4.f, 6.f}, {60.f, 70.f}, {5.f, 6.f}};
  out = blend.apply(active, t0 + std::chrono::milliseconds(50));
  expect_near("activation blend follows latest q target", out.command.q[0], 2.f);
  expect_near("activation blend follows latest kp target", out.command.kp[0], 40.f);
  expect_near("activation blend follows latest kd target", out.command.kd[0], 3.f);
  out = blend.apply(active, t0 + std::chrono::milliseconds(100));
  expect_true("activation blend completes at duration", out.completed);
  expect_near("activation blend reaches active q", out.command.q[0], 4.f);
  expect_near("activation blend reaches active kp", out.command.kp[0], 60.f);
  expect_near("activation blend reaches active kd", out.command.kd[0], 5.f);

  ActivationBlend disabled(0.0);
  disabled.observe_hold(hold);
  out = disabled.apply(active, t0);
  expect_near("zero activation blend reaches active immediately", out.command.q[0], 4.f);
  expect_throw("activation blend rejects negative duration", [] { ActivationBlend invalid(-0.1); });
  expect_throw("activation blend rejects excessive duration", [] { ActivationBlend invalid(5.1); });
  expect_throw("activation blend rejects non-finite duration", [] {
    ActivationBlend invalid(std::numeric_limits<double>::infinity());
  });
}

void test_async_command_publisher() {
  using action_go::runtime::AsyncCommandPublisher;
  using action_go::runtime::JointCommand;

  const JointCommand hold{{0.f}, {20.f}, {1.f}};
  std::mutex samples_mutex;
  std::vector<bool> active_samples;
  std::vector<std::chrono::steady_clock::time_point> publish_times;
  AsyncCommandPublisher publisher(
      500, 0.0, hold,
      [&](const JointCommand& command, bool active) {
        std::lock_guard<std::mutex> lock(samples_mutex);
        expect_near("async publisher callback command", command.q[0], active ? 1.f : 0.f);
        active_samples.push_back(active);
        publish_times.push_back(std::chrono::steady_clock::now());
      });

  std::this_thread::sleep_for(std::chrono::milliseconds(12));
  size_t hold_count = 0;
  {
    std::lock_guard<std::mutex> lock(samples_mutex);
    hold_count = active_samples.size();
    expect_true("async publisher keeps hold alive during inference", hold_count >= 3);
    expect_true("async publisher does not activate before first target",
                std::none_of(active_samples.begin(), active_samples.end(),
                             [](bool active) { return active; }));
  }
  publisher.update({{1.f}, {30.f}, {2.f}});
  std::this_thread::sleep_for(std::chrono::milliseconds(12));
  publisher.stop_and_join();
  publisher.rethrow_if_failed();
  const auto metrics = publisher.metrics();
  expect_true("async publisher emits active command after update", [&] {
    std::lock_guard<std::mutex> lock(samples_mutex);
    return active_samples.size() > hold_count &&
           std::any_of(active_samples.begin() + static_cast<std::ptrdiff_t>(hold_count),
                       active_samples.end(), [](bool active) { return active; });
  }());
  expect_true("async publisher metrics count callbacks",
              metrics.publish_count == static_cast<int64_t>(publish_times.size()));

  AsyncCommandPublisher failing(
      500, 0.0, hold,
      [](const JointCommand&, bool) { throw std::runtime_error("injected publisher failure"); });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  failing.stop_and_join();
  expect_true("async publisher records callback failure", failing.failed());
  expect_throw("async publisher propagates callback failure",
               [&] { failing.rethrow_if_failed(); });
}


int g_failures = 0;

void fail(const std::string& name, const std::string& msg) {
  ++g_failures;
  std::cerr << "[runtime_hardening_test] FAIL " << name << ": " << msg << "\n";
}

void expect_true(const std::string& name, bool ok) {
  if (!ok) fail(name, "condition is false");
}

void expect_near(const std::string& name, float actual, float expected, float eps) {
  if (std::abs(actual - expected) > eps) {
    fail(name, "actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
  }
}

void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try {
    fn();
    fail(name, "expected exception");
  } catch (const std::exception&) {
  }
}

void test_command_safety_filter() {
  using namespace action_go::runtime;

  CommandSafetyConfig cfg;
  cfg.lowstate_timeout_sec = 0.2;
  cfg.kp_max = 100.f;
  cfg.kd_max = 10.f;
  CommandSafetyFilter filter(cfg);

  CommandSafetyState state;
  state.connected = true;
  state.stamp = std::chrono::steady_clock::now();
  state.seq = 7;
  state.q = {0.1f, -0.1f, 0.0f};

  CommandSafetyLimits limits;
  limits.lower = {-1.f, -0.5f, -2.f};
  limits.upper = {1.f, 0.5f, 2.f};

  const auto out = filter.filter({2.f, -2.f, 0.25f}, {150.f, -1.f, 50.f}, {20.f, 2.f, -3.f},
                                 state, limits, 3);
  expect_near("safety clamps q high", out.q[0], 1.f);
  expect_near("safety clamps q low", out.q[1], -0.5f);
  expect_near("safety keeps q in range", out.q[2], 0.25f);
  expect_near("safety clamps kp high", out.kp[0], 100.f);
  expect_near("safety clamps kp low", out.kp[1], 0.f);
  expect_near("safety clamps kd high", out.kd[0], 10.f);
  expect_near("safety clamps kd low", out.kd[2], 0.f);
  const auto hold = filter.filter(out.q, out.kp, out.kd, state, limits, 3);
  for (size_t index = 0; index < out.q.size(); ++index) {
    expect_near("handoff hold preserves filtered q", hold.q[index], out.q[index]);
    expect_near("handoff hold preserves filtered kp", hold.kp[index], out.kp[index]);
    expect_near("handoff hold preserves filtered kd", hold.kd[index], out.kd[index]);
  }

  expect_throw("safety rejects q nan", [&] {
    filter.filter({std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
                  state, limits, 3);
  });
  expect_throw("safety rejects q inf", [&] {
    filter.filter({std::numeric_limits<float>::infinity(), 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
                  state, limits, 3);
  });
  expect_throw("safety rejects gain nan", [&] {
    filter.filter({0.f, 0.f, 0.f}, {std::numeric_limits<float>::quiet_NaN(), 1.f, 1.f},
                  {1.f, 1.f, 1.f}, state, limits, 3);
  });
  expect_throw("safety rejects missing lowstate", [&] {
    auto missing = state;
    missing.connected = false;
    filter.filter({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, missing, limits, 3);
  });
  expect_throw("safety rejects stale lowstate", [&] {
    auto stale = state;
    stale.stamp = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    filter.filter({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f}, stale, limits, 3);
  });

  auto stale = state;
  stale.stamp = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  const auto emergency = filter.filter({0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},
                                       stale, limits, 3, true);
  expect_true("safety allows stale emergency publish", emergency.q.size() == 3);

  CommandSafetyConfig parity_cfg;
  parity_cfg.profile = "simulation_parity";
  parity_cfg.joint_limit_enabled = false;
  parity_cfg.max_position_step_enabled = false;
  CommandSafetyFilter parity_filter(parity_cfg);
  const auto parity = parity_filter.filter({2.f, -2.f, 0.25f}, {10.f, 10.f, 10.f},
                                           {1.f, 1.f, 1.f}, state, limits, 3);
  expect_near("simulation parity preserves q above physical limit", parity.q[0], 2.f);
  expect_near("simulation parity preserves q below physical limit", parity.q[1], -2.f);
}

void test_pd_torque_filter() {
  using namespace action_go::runtime;
  CommandSafetyConfig config;
  config.preserve_pd_torque = true;
  CommandSafetyFilter filter(config);
  CommandSafetyState state;
  state.connected = true;
  state.stamp = std::chrono::steady_clock::now();
  state.q = {0.9f, -0.9f, 0.2f, 0.f};
  state.dq = {0.5f, -0.5f, 0.2f, 3.f};
  CommandSafetyLimits limits;
  limits.lower = {-1.f, -1.f, -1.f, -1.f};
  limits.upper = {1.f, 1.f, 1.f, 1.f};
  limits.torque = {10.f, 10.f, 10.f, 2.f};
  const std::vector<float> requested{4.f, -4.f, 0.25f, 0.f};
  const auto output = filter.filter(requested, {20.f, 20.f, 20.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, state, limits, 4);
  for (size_t joint = 0; joint < requested.size(); ++joint) {
    expect_true("PD compensation keeps physical target limits", output.q[joint] >= limits.lower[joint] && output.q[joint] <= limits.upper[joint]);
    const float actual = output.kp[joint] * (output.q[joint] - state.q[joint]) - output.kd[joint] * state.dq[joint] + output.tau[joint];
    const float expected = std::clamp(output.kp[joint] * (requested[joint] - state.q[joint]) - output.kd[joint] * state.dq[joint], -limits.torque[joint], limits.torque[joint]);
    expect_near("final LowCmd preserves saturated PD torque", actual, expected);
  }
  expect_true("out-of-range climbing target uses feedforward", output.tau[0] > 0.f && output.tau[1] < 0.f);
  expect_near("ordinary target requires no feedforward", output.tau[2], 0.f);
  auto fast = state;
  fast.dq[0] = 100.f;
  const auto saturated = filter.filter({10.f, -4.f, 0.25f, 0.f}, {20.f, 20.f, 20.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, fast, limits, 4);
  expect_near("bounded feedforward when PD representation is impossible", saturated.tau[0], 10.f);
  expect_near("unrepresentable PD disables proportional term", saturated.kp[0], 0.f);
  expect_near("unrepresentable PD disables damping term", saturated.kd[0], 0.f);
  expect_throw("PD compensation rejects nonfinite velocity", [&] {
    auto invalid = state;
    invalid.dq[0] = std::numeric_limits<float>::quiet_NaN();
    filter.filter(requested, {20.f, 20.f, 20.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, invalid, limits, 4);
  });
  expect_throw("PD compensation requires torque limits", [&] {
    filter.filter(requested, {20.f, 20.f, 20.f, 0.f}, {1.f, 1.f, 1.f, 1.f}, state, {}, 4);
  });
}

void test_prearm_command() {
  const auto command = action_go::runtime::validated_prearm_command_values();
  expect_true("prearm command finite",
              std::isfinite(command.position) && std::isfinite(command.velocity) &&
              std::isfinite(command.kp) && std::isfinite(command.kd) &&
              std::isfinite(command.torque));
  expect_true("prearm command fixed damping",
              command.position == 0.0f && command.kp == 0.0f &&
              command.kd > 0.0f && command.kd <= 20.0f && command.torque == 0.0f);
}

std::vector<uint8_t> make_npy(const std::string& descr, const std::string& shape,
                              const std::vector<uint8_t>& payload,
                              size_t extra_prefix = 0) {
  std::string header = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
  const size_t base = 10;
  const size_t padding = (16 - ((base + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  std::vector<uint8_t> bytes(extra_prefix, 0x7f);
  bytes.push_back(0x93);
  bytes.insert(bytes.end(), {'N', 'U', 'M', 'P', 'Y', 1, 0});
  bytes.push_back(static_cast<uint8_t>(header.size() & 0xff));
  bytes.push_back(static_cast<uint8_t>((header.size() >> 8) & 0xff));
  bytes.insert(bytes.end(), header.begin(), header.end());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  if (extra_prefix != 0) bytes.erase(bytes.begin(), bytes.begin() + extra_prefix);
  return bytes;
}

void test_npy_reader() {
  using namespace action_go::runtime::npy;
  const std::vector<double> doubles{1.25, -2.5};
  std::vector<uint8_t> payload(doubles.size() * sizeof(double) + 1, 0xaa);
  std::memcpy(payload.data() + 1, doubles.data(), doubles.size() * sizeof(double));
  auto bytes = make_npy("<f8", "(2,)",
                        std::vector<uint8_t>(payload.begin() + 1, payload.end()));
  const auto values = to_float_vector(bytes, "unaligned-f8");
  expect_true("npy f8 count", values.size() == 2);
  expect_near("npy f8 first", values[0], 1.25f);
  expect_near("npy f8 second", values[1], -2.5f);

  expect_throw("npy rejects trailing payload", [&] {
    auto trailing = bytes;
    trailing.push_back(0);
    (void)to_float_vector(trailing, "trailing");
  });
  expect_throw("npy rejects big endian", [&] {
    (void)to_float_vector(make_npy(">f8", "(2,)",
                                  std::vector<uint8_t>(payload.begin() + 1, payload.end())),
                          "big-endian");
  });
  expect_throw("npy rejects shape overflow", [&] {
    const auto metadata = parse_header(
        make_npy("<f4", "(18446744073709551615,2)", {}));
    (void)element_count(metadata.shape);
  });
}

std::string signed_udp_message(const std::string& key, const std::string& nonce,
                               int64_t issued_at_ms, int64_t execute_at_ms,
                               const std::string& type = "action",
                               const std::string& action = "kick_right") {
  const std::string key_id = "group-a";
  const std::string payload = "1\n" + type + "\n" + action + "\n" +
      std::to_string(issued_at_ms) + "\n" + std::to_string(execute_at_ms) + "\n" +
      nonce + "\n" + key_id;
  unsigned char digest[EVP_MAX_MD_SIZE]{};
  unsigned int digest_length = 0;
  HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()),
       reinterpret_cast<const unsigned char*>(payload.data()), payload.size(),
       digest, &digest_length);
  std::ostringstream signature;
  signature << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_length; ++i) {
    signature << std::setw(2) << static_cast<int>(digest[i]);
  }
  nlohmann::json message{{"version", 1}, {"type", type}, {"action", action},
                         {"issued_at_ms", issued_at_ms}, {"execute_at_ms", execute_at_ms},
                         {"nonce", nonce}, {"key_id", key_id},
                         {"signature", signature.str()}};
  return message.dump();
}

void test_network_command_authentication() {
  using namespace action_go::runtime;
  expect_true("UDP defaults loopback", is_loopback_bind_address("127.0.0.1"));
  expect_true("UDP empty bind is not loopback", !is_loopback_bind_address(""));
  expect_true("UDP IPv4 loopback range", is_loopback_bind_address("127.42.0.9"));
  expect_true("UDP IPv6 wildcard is not loopback", !is_loopback_bind_address("::"));
  expect_true("UDP wildcard not loopback", !is_loopback_bind_address("0.0.0.0"));
  const auto legacy = parse_legacy_network_command("ACTION kick_right delay_ms=25");
  expect_true("legacy action parsed", legacy.type == NetworkCommandType::kAction &&
              legacy.action == "kick_right" && legacy.delay_ms == 25);
  expect_throw("legacy rejects unknown field", [&] {
    (void)parse_legacy_network_command("ACTION kick_right command=/bin/sh");
  });

  const std::string key(32, 'k');
  AuthenticatedNetworkCommandParser::Config config;
  config.key = key;
  config.key_id = "group-a";
  AuthenticatedNetworkCommandParser parser(config);
  const int64_t now = 1700000000000;
  const std::string nonce = "nonce-1234567890";
  const std::string message = signed_udp_message(key, nonce, now, now + 100);
  const auto signed_command = parser.parse(message, now, true);
  expect_true("signed UDP accepted", signed_command.action == "kick_right" &&
              signed_command.execute_at_ms == now + 100);
  expect_throw("signed UDP replay rejected", [&] { (void)parser.parse(message, now, true); });
  expect_throw("signed UDP bad signature rejected", [&] {
    auto bad = nlohmann::json::parse(signed_udp_message(key, "nonce-abcdefghij", now, now));
    bad["signature"] = std::string(64, '0');
    (void)parser.parse(bad.dump(), now, true);
  });
  expect_throw("signed UDP expired rejected", [&] {
    const auto old = signed_udp_message(key, "nonce-old-123456", now - 20000, now - 19000);
    (void)parser.parse(old, now, true);
  });
  expect_throw("signed UDP future rejected", [&] {
    const auto future = signed_udp_message(key, "nonce-future-1234", now + 70000, now + 70000);
    (void)parser.parse(future, now, true);
  });
  expect_throw("signed UDP missing action field rejected", [&] {
    auto missing = nlohmann::json::parse(
        signed_udp_message(key, "nonce-missing-act", now, now));
    missing.erase("action");
    (void)parser.parse(missing.dump(), now, true);
  });
  expect_throw("signed UDP rejects action on stop command", [&] {
    const auto invalid = signed_udp_message(
        key, "nonce-stop-action1", now, now, "emergency", "kick_right");
    (void)parser.parse(invalid, now, true);
  });
  AuthenticatedNetworkCommandParser::Config small_config = config;
  small_config.replay_capacity = 1;
  AuthenticatedNetworkCommandParser small_parser(small_config);
  const auto first_nonce = signed_udp_message(key, "nonce-capacity-001", now, now);
  const auto second_nonce = signed_udp_message(key, "nonce-capacity-002", now, now);
  (void)small_parser.parse(first_nonce, now, true);
  expect_throw("signed UDP full replay cache rejects new command", [&] {
    (void)small_parser.parse(second_nonce, now, true);
  });
  expect_throw("signed UDP full replay cache still rejects replay", [&] {
    (void)small_parser.parse(first_nonce, now, true);
  });

  const auto key_path = std::filesystem::temp_directory_path() /
      ("actiongo_udp_key_" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  {
    std::ofstream output(key_path);
    output << key;
  }
  std::filesystem::permissions(key_path, std::filesystem::perms::owner_read |
      std::filesystem::perms::owner_write, std::filesystem::perm_options::replace);
  expect_true("UDP private key file accepted", load_udp_auth_key_file(key_path.string()) == key);
  std::filesystem::permissions(key_path, std::filesystem::perms::owner_read |
      std::filesystem::perms::owner_write | std::filesystem::perms::others_read,
      std::filesystem::perm_options::replace);
  expect_throw("UDP public key file rejected", [&] {
    (void)load_udp_auth_key_file(key_path.string());
  });
  std::filesystem::permissions(key_path, std::filesystem::perms::owner_read |
      std::filesystem::perms::owner_write | std::filesystem::perms::group_read,
      std::filesystem::perm_options::replace);
  expect_throw("UDP group-readable key file rejected", [&] {
    (void)load_udp_auth_key_file(key_path.string());
  });
  std::filesystem::remove(key_path);

  BoundedNetworkCommandQueue queue(2);
  expect_true("UDP queue accepts first action", queue.push(parse_legacy_network_command("ACTION first")));
  expect_true("UDP queue accepts second action", queue.push(parse_legacy_network_command("ACTION second")));
  expect_true("UDP queue rejects flooded action", !queue.push(parse_legacy_network_command("ACTION third")));
  expect_true("UDP priority command displaces queued action",
              queue.push(parse_legacy_network_command("damping")) && queue.size() == 2);
  NetworkCommand queued;
  expect_true("UDP priority command dequeues first",
              queue.pop(queued) && queued.type == NetworkCommandType::kDampingExit);
  expect_true("UDP queue keeps emergency ahead of priority commands",
              queue.push(parse_legacy_network_command("walkrun")) &&
              queue.push(parse_legacy_network_command("emergency")) &&
              queue.pop(queued) && queued.type == NetworkCommandType::kEmergency);
}

void test_reference_endpoint_boundary() {
  using action_go::runtime::is_local_reference_endpoint;
  expect_true("reference IPC endpoint is local", is_local_reference_endpoint("ipc:///tmp/actiongo-reference"));
  expect_true("reference inproc endpoint is local", is_local_reference_endpoint("inproc://reference"));
  expect_true("reference IPv4 loopback endpoint is local", is_local_reference_endpoint("tcp://127.0.0.1:15111"));
  expect_true("reference IPv4 loopback range is local", is_local_reference_endpoint("tcp://127.42.0.9:15111"));
  expect_true("reference IPv6 loopback endpoint is local", is_local_reference_endpoint("tcp://[::1]:15111"));
  expect_true("reference wildcard endpoint is remote", !is_local_reference_endpoint("tcp://0.0.0.0:15111"));
  expect_true("reference remote endpoint is remote", !is_local_reference_endpoint("tcp://192.0.2.1:15111"));
  expect_true("reference DNS endpoint is rejected", !is_local_reference_endpoint("tcp://robot.local:15111"));
  expect_true("reference malformed endpoint is rejected", !is_local_reference_endpoint("tcp://[::1"));
}

void test_onnx_runner() {
  using namespace action_go::runtime;

#ifndef ACTION_GO_TEST_ONNX_MODEL
  throw std::runtime_error("ACTION_GO_TEST_ONNX_MODEL is not defined");
#else
  const std::filesystem::path model(ACTION_GO_TEST_ONNX_MODEL);
  if (!std::filesystem::exists(model)) {
    throw std::runtime_error("test ONNX model not found: " + model.string());
  }

  OnnxRunner runner;
  runner.load(model, 1);
  runner.require_inputs({"obs"});
  runner.require_outputs({"actions"});
  expect_throw("onnx missing input", [&] { runner.require_inputs({"not_an_input"}); });
  expect_throw("onnx missing output", [&] { runner.require_outputs({"not_an_output"}); });

  const int obs_dim = runner.inferred_obs_dim(98);
  std::vector<float> obs(static_cast<size_t>(obs_dim), 0.f);
  const auto action = runner.run_single(obs, "obs", "actions", 29);
  expect_true("onnx action size", action.size() == 29);

  obs[0] = std::numeric_limits<float>::quiet_NaN();
  expect_throw("onnx rejects nan input", [&] { runner.run_single(obs, "obs", "actions", 29); });
  obs[0] = 0.f;

  expect_throw("onnx rejects output too small", [&] { runner.run_single(obs, "obs", "actions", 30); });
  expect_throw("onnx rejects output too large", [&] { runner.run_single(obs, "obs", "actions", 28); });

  OnnxRunner auto_runner;
  OnnxRunner::LoadOptions auto_options;
  auto_options.execution_provider = "auto";
  auto_runner.load(model, auto_options);
  expect_true("onnx auto selects provider", !auto_runner.selected_execution_provider().empty());
  OnnxRunner bad_runner;
  OnnxRunner::LoadOptions bad_options;
  bad_options.execution_provider = "not-a-provider";
  expect_throw("onnx rejects unknown provider", [&] { bad_runner.load(model, bad_options); });
#endif
}

void test_realtime_input_stale_gate() {
  using namespace action_go::runtime;

  RealtimeInputStaleGate gate;
  const auto t0 = std::chrono::steady_clock::now();
  auto d = gate.decide(SampleFreshness::kFresh, StaleBehavior::kHoldThenStop, 0.5, t0);
  expect_true("fresh sample is used", d.action == StaleDecisionAction::kUseSample);

  d = gate.decide(SampleFreshness::kStale, StaleBehavior::kHoldThenStop, 0.5, t0);
  expect_true("first stale holds", d.action == StaleDecisionAction::kHoldLast);
  expect_true("first stale event marked", d.new_stale_event);

  d = gate.decide(SampleFreshness::kStale, StaleBehavior::kHoldThenStop, 0.5,
                  t0 + std::chrono::milliseconds(200));
  expect_true("short stale keeps holding", d.action == StaleDecisionAction::kHoldLast);
  expect_true("continued stale is not new", !d.new_stale_event);

  d = gate.decide(SampleFreshness::kMissing, StaleBehavior::kHoldThenStop, 0.5,
                  t0 + std::chrono::milliseconds(600));
  expect_true("timeout stale stops", d.action == StaleDecisionAction::kStop);

  d = gate.decide(SampleFreshness::kMissing, StaleBehavior::kDefault, 0.5, t0);
  expect_true("default stale behavior uses default", d.action == StaleDecisionAction::kUseDefault);
  d = gate.decide(SampleFreshness::kMissing, StaleBehavior::kStop, 0.5, t0);
  expect_true("stop stale behavior stops", d.action == StaleDecisionAction::kStop);
}

void test_policy_contract() {
  using namespace action_go::runtime;

  const PolicyContract* single = contract_for_template("single_actor_state");
  expect_true("single_actor_state contract exists", single != nullptr);
  expect_true("single_actor_state player", single && single->player_kind == GenericPlayerKind::kPolicy);
  expect_true("single_actor_state variant", variant_allowed_for_template("single_actor_state", "bfm_zero"));

  const PolicyContract* mimic = contract_for_template("embedded_mimic_time");
  expect_true("embedded_mimic_time contract exists", mimic != nullptr);
  expect_true("embedded_mimic_time variant", mimic && variant_allowed_for_template("embedded_mimic_time", "holosoma"));

  const PolicyContract* motion = contract_for_template("motion_conditioned_tracking");
  expect_true("motion_conditioned_tracking player", motion && motion->player_kind == GenericPlayerKind::kMotionTracking);
  const PolicyContract* motion_alias = contract_for_template("external_reference_tracking");
  expect_true("external_reference_tracking alias", motion_alias && motion && motion_alias->id == motion->id);
  expect_true("twist motion player", variant_allowed_for_player("twist", GenericPlayerKind::kMotionTracking));
  expect_true("motion not policy player", !template_allowed_for_player("motion_conditioned_tracking", GenericPlayerKind::kPolicy));

  const PolicyVariantContract* humanoid_gpt = variant_contract("humanoid_gpt");
  expect_true("humanoid_gpt extension exists", humanoid_gpt != nullptr);
  expect_true("humanoid_gpt not core paper coverage", humanoid_gpt && !humanoid_gpt->core_paper_coverage);
  expect_true("sonic motion player", variant_allowed_for_player("sonic", GenericPlayerKind::kMotionTracking));

  const PolicyContract* depth = contract_for_template("depth_conditioned_actor");
  expect_true("depth sensor player", depth && depth->player_kind == GenericPlayerKind::kSensor);
  const PolicyContract* object = contract_for_template("object_conditioned_actor");
  expect_true("object sensor player", object && object->player_kind == GenericPlayerKind::kSensor);
  const PolicyContract* composite = contract_for_template("composite_motion_policy");
  expect_true("composite player", composite && composite->player_kind == GenericPlayerKind::kComposite);

  expect_true("policy templates format includes single",
              format_supported_templates(GenericPlayerKind::kPolicy).find("single_actor_state") != std::string::npos);
  expect_true("motion variants format includes protomotions",
              format_supported_variants(GenericPlayerKind::kMotionTracking).find("protomotions") != std::string::npos);
  expect_true("template variants format includes unitree",
              format_supported_variants_for_template("single_actor_state").find("unitree_rl_mjlab") != std::string::npos);
}

void test_policy_protocol_constraints() {
  using namespace action_go::runtime;

  const std::vector<std::string> variants{
      "bfm_zero", "unitree_rl_mjlab", "beyondmimic", "holosoma",
      "protomotions", "twist", "humanoid_gpt", "instinct_parkour",
      "sonic", "instinct", "php", "pbfm", "steadytray", "omnixtreme", "supermimic", "uni_supermimic", "allmimic"};
  for (const std::string& variant : variants) {
    const auto* contract = variant_contract(variant);
    const auto* protocol = protocol_constraint_for_variant(variant);
    expect_true("protocol exists for " + variant, protocol != nullptr);
    expect_true("contract exists for " + variant, contract != nullptr);
    if (protocol) {
      expect_true("protocol is G1-specific for " + variant, protocol->unitree_g1_specific);
      expect_true("protocol motor count for " + variant, protocol->motor_count == kUnitreeG1ProtocolMotorCount);
      expect_true("protocol obs dim positive for " + variant, protocol->obs_dim > 0);
      expect_true("protocol action dim positive for " + variant, protocol->action_dim > 0);
    }
  }

  const auto* twist = protocol_constraint_for_variant("twist");
  expect_true("twist protocol exists", twist != nullptr);
  if (twist) {
    expect_true("twist mimic dim", twist->twist_mimic_dim == kTwistProtocolMimicDim);
    expect_true("twist wrist dim", twist->twist_wrist_dim == kTwistProtocolWristDim);
    expect_true("twist obs dim", twist->twist_input_dim == kTwistProtocolObsDim);
  }

  const auto* humanoid_gpt = protocol_constraint_for_variant("humanoid_gpt");
  expect_true("humanoid_gpt protocol exists", humanoid_gpt != nullptr);
  if (humanoid_gpt) {
    expect_true("humanoid_gpt qpos dim", humanoid_gpt->humanoid_gpt_qpos_dim == kHumanoidGptProtocolQposDim);
    expect_true("humanoid_gpt qvel dim", humanoid_gpt->humanoid_gpt_qvel_dim == kHumanoidGptProtocolQvelDim);
    expect_true("humanoid_gpt keypoints", humanoid_gpt->humanoid_gpt_keypoint_count == kHumanoidGptProtocolKeypointCount);
  }

  const auto* sonic = protocol_constraint_for_variant("sonic");
  expect_true("sonic protocol exists", sonic != nullptr);
  if (sonic) {
    expect_true("sonic token dim", sonic->latent_dim == kSonicTokenDim);
    expect_true("sonic decoder obs dim", sonic->obs_dim == kSonicDecoderObsDim);
    expect_true("sonic has three sessions", sonic->sessions.size() == 3);
  }
}

void test_allmimic_policy_helpers() {
  using namespace action_go::runtime;
  AllMimicController controller;
  std::vector<float> reference_weight(29, 1.f);
  std::fill_n(reference_weight.begin(), 12, 0.f);
  controller.configure(std::vector<float>(29, 0.25f), std::vector<float>(29, 0.5f),
                       std::vector<float>(29, 1.f), reference_weight);
  SuperMimicReferenceFrame reference;
  for (std::size_t frame = 0; frame < 10; ++frame) {
    reference.goal[frame * 73 + 3] = 1.f;
    reference.goal[frame * 73 + 7] = 1.f;
    for (std::size_t joint = 0; joint < 29; ++joint) reference.goal[frame * 73 + 9 + joint] = 0.75f;
  }
  controller.set_reference_frames({reference, reference});
  SuperMimicRobotFrame robot;
  robot.q.fill(0.5f);
  robot.quat = {0.70710678f, 0.f, 0.f, 0.70710678f};
  auto input = controller.build_input(robot);
  expect_true("AllMimic packed input dimension", input.size() == 1660);
  expect_near("AllMimic history uses 93 features", input[9 * 93], 0.25f);
  expect_near("AllMimic column-major rotated x axis", input[930 + 4], -1.f);
  expect_near("AllMimic column-major rotated y axis", input[930 + 6], 1.f);
  const auto target = controller.decode_action(std::vector<float>(29, 2.f));
  expect_near("AllMimic legs use absolute targets", target[0], 1.25f);
  expect_near("AllMimic upper body uses next reference residual", target[12], 1.75f);
  controller.accept_action(std::vector<float>(29, 2.f));
  input = controller.build_input(robot);
  expect_near("AllMimic stores raw previous action", input[9 * 93 + 64], 2.f);
  controller.reset(robot.quat, true);
  input = controller.build_input(robot);
  expect_near("AllMimic initial yaw alignment", input[930 + 3], 1.f);
  expect_near("AllMimic reset clears raw previous action", input[64], 0.f);
  expect_true("AllMimic supports tracking player", variant_allowed_for_player("allmimic", GenericPlayerKind::kMotionTracking));
  const auto* protocol = protocol_constraint_for_variant("allmimic");
  expect_true("AllMimic protocol input", protocol && protocol->obs_dim == 1660);
}

void test_supermimic_policy_helpers() {
  using namespace action_go::runtime;
  expect_true("Uni-SuperMimic uses motion tracking player",
              variant_allowed_for_player("uni_supermimic", GenericPlayerKind::kMotionTracking));
  expect_true("Uni-SuperMimic rejects state-only template",
              !variant_allowed_for_template("single_actor_state", "uni_supermimic"));
  const auto* protocol = protocol_constraint_for_variant("uni_supermimic");
  expect_true("Uni-SuperMimic packed input contract", protocol && protocol->obs_dim == 1720);
  expect_true("Uni-SuperMimic output contract", protocol && protocol->action_dim == 29);
  const auto rules = protocol_yaml_dim_rules_for_variant("uni_supermimic");
  expect_true("Uni-SuperMimic validates history and goal dimensions", rules.size() == 4);
  SuperMimicController controller;
  controller.configure(std::vector<float>(29, 0.25f), std::vector<float>(29, 0.5f));
  SuperMimicReferenceFrame first, last;
  first.goal[0] = 10.f;
  last.goal[0] = 20.f;
  controller.set_reference_frames({first, last});
  SuperMimicRobotFrame robot;
  robot.q.fill(0.5f);
  robot.dq[0] = 3.f;
  robot.gyro = {1.f, 2.f, 3.f};
  const auto initial = controller.build_input(robot);
  expect_true("supermimic packed dimension", initial.size() == 1720);
  expect_near("supermimic raw joint velocity", initial[29], 3.f);
  expect_near("supermimic pelvis gyro", initial[59], 2.f);
  expect_near("supermimic gravity", initial[63], -1.f);
  expect_near("supermimic relative rotation first column", initial[93], 1.f);
  expect_near("supermimic interleaved rotation second column", initial[96], 1.f);
  for (std::size_t frame = 0; frame < 10; ++frame) {
    expect_near("supermimic fills initial history", initial[99 * frame], 0.25f);
  }
  expect_near("supermimic first reference", initial[990], 10.f);
  std::vector<float> action(29, 4.f);
  const auto target = controller.decode_action(action);
  expect_near("supermimic unbounded raw action decoding", target[0], 2.25f);
  controller.accept_action(action);
  robot.q[0] = 1.f;
  const auto advanced = controller.build_input(robot);
  expect_near("supermimic oldest history remains", advanced[0], 0.25f);
  expect_near("supermimic newest history appended", advanced[891], 0.75f);
  expect_near("supermimic previous raw action", advanced[891 + 64], 4.f);
  expect_near("supermimic advances reference", advanced[990], 20.f);
  controller.accept_action(action);
  controller.accept_action(action);
  expect_true("supermimic reference clamps at end", controller.reference_index() == 1);
  expect_near("supermimic continues inference after end", controller.build_input(robot)[990], 20.f);
  robot.quat = {0.70710678f, 0.f, 0.f, 0.70710678f};
  controller.reset(robot.quat, true);
  const auto aligned = controller.build_input(robot);
  expect_near("supermimic initial yaw alignment", aligned[93], 1.f);
  expect_near("supermimic reset clears previous actions", aligned[64], 0.f);
  robot.quat.fill(0.f);
  expect_throw("supermimic rejects zero quaternion", [&] { controller.build_input(robot); });
  action[0] = std::numeric_limits<float>::quiet_NaN();
  expect_throw("supermimic rejects nonfinite actions", [&] { controller.accept_action(action); });
  std::istringstream malformed("0,1,2\n");
  expect_throw("supermimic rejects incomplete reference", [&] {
    SuperMimicController::read_reference_csv(malformed);
  });
}

void test_sonic_policy_helpers() {
  using namespace action_go::runtime;
  std::vector<float> hardware(kSonicMotorCount), defaults(kSonicMotorCount, 0.f), scales(kSonicMotorCount, 1.f);
  for (int i = 0; i < kSonicMotorCount; ++i) hardware[i] = static_cast<float>(i);
  const auto policy = sonic_robot_to_policy(hardware, defaults, false);
  for (int i = 0; i < kSonicMotorCount; ++i) {
    expect_near("sonic robot remap " + std::to_string(i), policy[i], hardware[sonic_policy_to_hardware()[i]]);
  }
  const std::vector<float> policy_action(policy.begin(), policy.end());
  const auto decoded = sonic_decode_target(policy_action, defaults, scales);
  for (int i = 0; i < kSonicMotorCount; ++i) {
    expect_near("sonic action remap " + std::to_string(i), decoded[i], hardware[i]);
  }

  std::vector<float> measured(kSonicMotorCount, -0.4f);
  std::vector<float> startup_default(kSonicMotorCount, 0.6f);
  const auto startup_mid = sonic_startup_target(measured, startup_default, 0.25f);
  expect_near("sonic startup uses latest measured pose", startup_mid[0], -0.15f);
  measured[0] = 0.2f;
  const auto startup_updated = sonic_startup_target(measured, startup_default, 0.25f);
  expect_near("sonic startup follows updated pose", startup_updated[0], 0.3f);
  expect_near("sonic startup alpha clamps low",
              sonic_startup_target(measured, startup_default, -1.f)[0], 0.2f);
  expect_near("sonic startup alpha clamps high",
              sonic_startup_target(measured, startup_default, 2.f)[0], 0.6f);

  SonicReferenceFrame reference_frame;
  SonicReferenceCache reference;
  reference.replace({reference_frame});
  SonicRobotFrame robot;
  SonicObservationBuilder builder;
  builder.reset(robot, reference_frame);
  builder.push_robot_frame(robot);
  const auto encoder_obs = builder.build_encoder_observation(reference, 0, kSonicReferenceFrames, kSonicReferenceStep);
  const auto decoder_obs = builder.build_decoder_observation(std::vector<float>(kSonicTokenDim, 0.f));
  expect_true("sonic encoder obs size", encoder_obs.size() == kSonicEncoderObsDim);
  expect_true("sonic decoder obs size", decoder_obs.size() == kSonicDecoderObsDim);
  for (size_t i = 584; i < 601; ++i) {
    expect_near("sonic inactive encoder slot " + std::to_string(i), encoder_obs[i], 0.f);
  }

  SonicReferenceFrame rotated_reference;
  rotated_reference.body_quat_wxyz = {0.70710678f, 0.f, 0.f, 0.70710678f};
  SonicReferenceCache rotated_cache;
  rotated_cache.replace({rotated_reference});
  SonicObservationBuilder rotated_builder;
  rotated_builder.reset(robot, reference_frame);
  rotated_builder.push_robot_frame(robot);
  const auto rotated_obs = rotated_builder.build_encoder_observation(
      rotated_cache, 0, kSonicReferenceFrames, kSonicReferenceStep);
  const std::array<float, 6> expected_rot6{0.f, -1.f, 1.f, 0.f, 0.f, 0.f};
  for (size_t i = 0; i < expected_rot6.size(); ++i) {
    expect_near("sonic row-major rotation6d " + std::to_string(i), rotated_obs[601 + i], expected_rot6[i], 1e-5f);
  }

  SonicObservationBuilder v11_builder;
  v11_builder.set_layout(SonicObservationLayout::kV11);
  v11_builder.reset(robot, reference_frame);
  SonicRobotFrame pitched_robot = robot;
  pitched_robot.base_quat_wxyz = {0.96592583f, 0.f, 0.25881905f, 0.f};
  v11_builder.push_robot_frame(pitched_robot);
  const auto v11_obs = v11_builder.build_encoder_observation(
      reference, 0, kSonicReferenceFrames, kSonicReferenceStep);
  expect_true("sonic v1.1 encoder obs size", v11_obs.size() == kSonicV11EncoderObsDim);
  const std::array<float, 6> identity_rot6{1.f, 0.f, 0.f, 1.f, 0.f, 0.f};
  for (size_t i = 0; i < identity_rot6.size(); ++i) {
    expect_near("sonic v1.1 heading-only orientation " + std::to_string(i),
                v11_obs[584 + i], identity_rot6[i], 1e-5f);
  }

  SonicObservationBuilder history_builder;
  history_builder.reset(robot, reference_frame);
  for (int frame = 1; frame <= kSonicHistoryLen; ++frame) {
    SonicRobotFrame history = robot;
    history.base_ang_vel[0] = static_cast<float>(frame);
    history_builder.push_robot_frame(history);
  }
  const auto history_obs = history_builder.build_decoder_observation(std::vector<float>(kSonicTokenDim, 0.f));
  expect_near("sonic history oldest first", history_obs[kSonicTokenDim], 1.f);
  expect_near("sonic history newest last", history_obs[kSonicTokenDim + 27], 10.f);

  history_builder.reset(robot, reference_frame);
  SonicRobotFrame first_control = robot;
  first_control.base_ang_vel[0] = 42.f;
  history_builder.push_robot_frame(first_control);
  const auto cold_control_obs = history_builder.build_decoder_observation(
      std::vector<float>(kSonicTokenDim, 0.f));
  for (int frame = 0; frame < kSonicHistoryLen - 1; ++frame) {
    expect_near("sonic reset zero-pads startup history " + std::to_string(frame),
                cold_control_obs[kSonicTokenDim + static_cast<size_t>(frame) * 3], 0.f);
  }
  expect_near("sonic first control frame is newest",
              cold_control_obs[kSonicTokenDim + (kSonicHistoryLen - 1) * 3], 42.f);

  SonicReferenceCache window;
  std::vector<SonicReferenceFrame> first_window(4), overlapping_window(4);
  for (int i = 0; i < 4; ++i) {
    first_window[i].frame_index = i;
    overlapping_window[i].frame_index = i + 2;
    overlapping_window[i].joint_pos[0] = 10.f + i;
  }
  window.replace(first_window);
  window.merge_tail(overlapping_window);
  expect_true("sonic overlapping window merged", window.size() == 6 && window.last_frame_index() == 5);
  expect_near("sonic overlapping frame replaced", window.frame_clamped(2).joint_pos[0], 10.f);

  std::vector<float> q(kSonicMotorCount), dq(kSonicMotorCount);
  for (int i = 0; i < kSonicMotorCount; ++i) { q[i] = 0.01f * i; dq[i] = -0.01f * i; }
  const std::array<float, 4> quat{1.f, 0.f, 0.f, 0.f};
  const int64_t frame_index = 7;
  const std::string header =
      "{\"v\":1,\"endian\":\"le\",\"count\":1,\"fields\":["
      "{\"name\":\"joint_pos\",\"dtype\":\"f32\",\"shape\":[1,29]},"
      "{\"name\":\"joint_vel\",\"dtype\":\"f32\",\"shape\":[1,29]},"
      "{\"name\":\"body_quat\",\"dtype\":\"f32\",\"shape\":[1,4]},"
      "{\"name\":\"frame_index\",\"dtype\":\"i64\",\"shape\":[1]}]}";
  std::vector<uint8_t> packet;
  const std::string topic = "pose";
  packet.insert(packet.end(), topic.begin(), topic.end());
  packet.resize(topic.size() + kSonicZmqHeaderSize, 0);
  std::memcpy(packet.data() + topic.size(), header.data(), header.size());
  auto append_bytes = [&](const void* data, size_t bytes) {
    const auto* begin = static_cast<const uint8_t*>(data);
    packet.insert(packet.end(), begin, begin + bytes);
  };
  append_bytes(q.data(), q.size() * sizeof(float));
  append_bytes(dq.data(), dq.size() * sizeof(float));
  append_bytes(quat.data(), quat.size() * sizeof(float));
  append_bytes(&frame_index, sizeof(frame_index));
  std::string layout_signature;
  const auto parsed = parse_sonic_zmq_v1_packet(packet, topic, &layout_signature);
  expect_true("sonic zmq frame count", parsed.size() == 1);
  expect_true("sonic zmq frame index", parsed.size() == 1 && parsed[0].frame_index == frame_index);
  expect_near("sonic zmq q", parsed[0].joint_pos[10], q[10]);
  expect_true("sonic explicit frame index", parsed[0].has_explicit_frame_index);
  expect_true("sonic layout signature", layout_signature.find("joint_pos:f32") != std::string::npos);
  expect_throw("sonic zmq requires body quat", [&] {
    std::string broken = header;
    const auto pos = broken.find("body_quat");
    broken.replace(pos, std::strlen("body_quat"), "bad_field");
    auto bad_packet = packet;
    std::fill(bad_packet.begin() + topic.size(), bad_packet.begin() + topic.size() + kSonicZmqHeaderSize, 0);
    std::memcpy(bad_packet.data() + topic.size(), broken.data(), broken.size());
    (void)parse_sonic_zmq_v1_packet(bad_packet, topic);
  });
  expect_throw("sonic zmq requires joint position", [&] {
    std::string broken = header;
    const auto pos = broken.find("joint_pos");
    broken.replace(pos, std::strlen("joint_pos"), "bad_joint");
    auto bad_packet = packet;
    std::fill(bad_packet.begin() + topic.size(), bad_packet.begin() + topic.size() + kSonicZmqHeaderSize, 0);
    std::memcpy(bad_packet.data() + topic.size(), broken.data(), broken.size());
    (void)parse_sonic_zmq_v1_packet(bad_packet, topic);
  });
  expect_throw("sonic zmq rejects trailing bytes", [&] {
    auto bad_packet = packet;
    bad_packet.push_back(0);
    (void)parse_sonic_zmq_v1_packet(bad_packet, topic);
  });

  constexpr size_t pico_frames = 5;
  std::vector<float> smpl_joints(pico_frames * kSonicSmplJointCount * 3);
  std::vector<float> smpl_pose(pico_frames * kSonicSmplPoseCount * 3);
  std::vector<double> pico_q(pico_frames * kSonicMotorCount);
  std::vector<double> pico_dq(pico_frames * kSonicMotorCount);
  std::vector<float> pico_quat(pico_frames * 4, 0.f);
  std::vector<int64_t> pico_indices(pico_frames);
  for (size_t i = 0; i < pico_frames; ++i) {
    pico_quat[i * 4] = 1.f;
    pico_indices[i] = static_cast<int64_t>(100 + i);
  }
  smpl_joints[17] = 0.25f;
  pico_q[23] = 0.4;
  const bool toggle = false;
  const std::string pico_header =
      "{\"v\":3,\"endian\":\"le\",\"count\":1,\"fields\":["
      "{\"name\":\"smpl_pose\",\"dtype\":\"f32\",\"shape\":[5,21,3]},"
      "{\"name\":\"smpl_joints\",\"dtype\":\"f32\",\"shape\":[5,24,3]},"
      "{\"name\":\"body_quat_w\",\"dtype\":\"f32\",\"shape\":[5,4]},"
      "{\"name\":\"joint_pos\",\"dtype\":\"f64\",\"shape\":[5,29]},"
      "{\"name\":\"joint_vel\",\"dtype\":\"f64\",\"shape\":[5,29]},"
      "{\"name\":\"frame_index\",\"dtype\":\"i64\",\"shape\":[5]},"
      "{\"name\":\"toggle_data_collection\",\"dtype\":\"bool\",\"shape\":[1]}]}";
  std::vector<uint8_t> pico_packet(topic.begin(), topic.end());
  pico_packet.resize(topic.size() + kSonicZmqHeaderSize, 0);
  std::memcpy(pico_packet.data() + topic.size(), pico_header.data(), pico_header.size());
  auto append_pico = [&](const void* data, size_t bytes) {
    const auto* begin = static_cast<const uint8_t*>(data);
    pico_packet.insert(pico_packet.end(), begin, begin + bytes);
  };
  append_pico(smpl_pose.data(), smpl_pose.size() * sizeof(float));
  append_pico(smpl_joints.data(), smpl_joints.size() * sizeof(float));
  append_pico(pico_quat.data(), pico_quat.size() * sizeof(float));
  append_pico(pico_q.data(), pico_q.size() * sizeof(double));
  append_pico(pico_dq.data(), pico_dq.size() * sizeof(double));
  append_pico(pico_indices.data(), pico_indices.size() * sizeof(int64_t));
  append_pico(&toggle, sizeof(toggle));
  const auto pico_parsed = parse_sonic_zmq_packet(pico_packet, topic);
  expect_true("sonic pico v3 frames", pico_parsed.size() == pico_frames);
  expect_true("sonic pico v3 encoder mode", pico_parsed[0].encoder_mode == 2);
  expect_near("sonic pico v3 smpl", pico_parsed[0].smpl_joints[17], 0.25f);
  expect_near("sonic pico v3 wrist", pico_parsed[0].joint_pos[23], 0.4f);
  expect_true("sonic pico v3 frame index", pico_parsed[4].frame_index == 104);

  SonicReferenceCache pico_cache;
  pico_cache.replace(pico_parsed);
  SonicObservationBuilder pico_builder;
  pico_builder.reset(robot, pico_parsed.front());
  pico_builder.push_robot_frame(robot);
  const auto pico_encoder_obs = pico_builder.build_encoder_observation(
      pico_cache, 0, kSonicReferenceFrames, kSonicReferenceStep);
  expect_near("sonic pico encoder mode slot", pico_encoder_obs[0], 2.f);
  expect_near("sonic pico smpl slot", pico_encoder_obs[922 + 17], 0.25f);
  expect_near("sonic pico wrist slot", pico_encoder_obs[1702], 0.4f);

  SonicObservationBuilder pico_v11_builder;
  pico_v11_builder.set_layout(SonicObservationLayout::kV11);
  pico_v11_builder.reset(robot, pico_parsed.front());
  pico_v11_builder.push_robot_frame(robot);
  const auto pico_v11_obs = pico_v11_builder.build_encoder_observation(
      pico_cache, 0, kSonicReferenceFrames, kSonicReferenceStep);
  expect_true("sonic v1.1 pico obs size", pico_v11_obs.size() == kSonicV11EncoderObsDim);
  expect_near("sonic v1.1 pico smpl slot", pico_v11_obs[911 + 17], 0.25f);
  expect_near("sonic v1.1 pico wrist slot", pico_v11_obs[1691], 0.4f);
}

void test_sonic_planner_helpers() {
  using namespace action_go::runtime;

  auto append_bytes = [](std::vector<uint8_t>& packet, const void* data, size_t bytes) {
    const auto* begin = static_cast<const uint8_t*>(data);
    packet.insert(packet.end(), begin, begin + bytes);
  };
  auto make_packet = [&](const std::string& topic, const nlohmann::json& fields,
                         const std::function<void(std::vector<uint8_t>&)>& append_payload) {
    const std::string header = nlohmann::json{
        {"v", 1}, {"endian", "le"}, {"count", 1}, {"fields", fields}}.dump();
    std::vector<uint8_t> packet(topic.begin(), topic.end());
    packet.resize(topic.size() + kSonicZmqHeaderSize, 0);
    std::memcpy(packet.data() + topic.size(), header.data(), header.size());
    append_payload(packet);
    return packet;
  };

  const nlohmann::json planner_fields = nlohmann::json::array({
      {{"name", "mode"}, {"dtype", "i32"}, {"shape", {1}}},
      {{"name", "movement"}, {"dtype", "f32"}, {"shape", {3}}},
      {{"name", "facing"}, {"dtype", "f32"}, {"shape", {3}}},
      {{"name", "speed"}, {"dtype", "f32"}, {"shape", {1}}},
      {{"name", "height"}, {"dtype", "f32"}, {"shape", {1}}}});
  const int32_t mode = 8;
  const std::array<float, 3> movement{0.5f, -0.25f, 0.f};
  const std::array<float, 3> facing{0.f, 1.f, 0.f};
  const float speed = 0.7f;
  const float height = -1.f;
  const auto planner_packet = make_packet("planner", planner_fields, [&](std::vector<uint8_t>& packet) {
    append_bytes(packet, &mode, sizeof(mode));
    append_bytes(packet, movement.data(), sizeof(movement));
    append_bytes(packet, facing.data(), sizeof(facing));
    append_bytes(packet, &speed, sizeof(speed));
    append_bytes(packet, &height, sizeof(height));
  });
  const auto planner_command = parse_sonic_planner_packet(planner_packet);
  expect_true("sonic planner mode parsed", planner_command.mode == 8);
  expect_near("sonic planner movement parsed", planner_command.movement[1], -0.25f);
  expect_true("sonic crawling cadence", sonic_planner_replan_period_sec(8) == 0.2);
  expect_true("sonic squat is static", sonic_planner_mode_is_static(4));
  expect_true("sonic walk is dynamic", !sonic_planner_mode_is_static(2));
  expect_true("sonic V2 mode name", std::string(sonic_planner_mode_name(26)) == "scare_walk");

  expect_throw("sonic planner rejects mode 27", [&] {
    auto bad = planner_packet;
    const int32_t invalid = 27;
    std::memcpy(bad.data() + 7 + kSonicZmqHeaderSize, &invalid, sizeof(invalid));
    (void)parse_sonic_planner_packet(bad);
  });

  const nlohmann::json command_fields = nlohmann::json::array({
      {{"name", "start"}, {"dtype", "u8"}, {"shape", {1}}},
      {{"name", "stop"}, {"dtype", "u8"}, {"shape", {1}}},
      {{"name", "planner"}, {"dtype", "u8"}, {"shape", {1}}}});
  const auto command_packet = make_packet("command", command_fields, [&](std::vector<uint8_t>& packet) {
    const std::array<uint8_t, 3> values{1, 0, 1};
    append_bytes(packet, values.data(), values.size());
  });
  const auto control = parse_sonic_command_packet(command_packet);
  expect_true("sonic planner control parsed", control.start && !control.stop && control.planner);

  SonicPlanner planner;
  std::vector<float> hardware_q(kSonicMotorCount);
  for (int i = 0; i < kSonicMotorCount; ++i) hardware_q[i] = static_cast<float>(i) * 0.01f;
  const auto initial = planner.initial_context(hardware_q);
  expect_near("sonic planner initial height", initial[2], 0.788740f);
  expect_near("sonic planner initial hardware order", initial[7 + 17], hardware_q[17]);

  std::vector<float> qpos(kSonicPlannerMaxFrames * kSonicPlannerQposDim, 0.f);
  for (int frame = 0; frame < kSonicPlannerMaxFrames; ++frame) {
    qpos[frame * kSonicPlannerQposDim + 2] = 0.8f;
    qpos[frame * kSonicPlannerQposDim + 3] = 1.f;
    for (int hardware = 0; hardware < kSonicMotorCount; ++hardware) {
      qpos[frame * kSonicPlannerQposDim + 7 + hardware] =
          static_cast<float>(frame) + static_cast<float>(hardware) * 0.01f;
    }
  }
  const auto resampled = planner.resample_output(qpos, 24);
  expect_true("sonic planner 30 to 50 frame count", resampled.size() == 40);
  expect_near("sonic planner output remap", resampled[0].joint_pos[7],
              static_cast<float>(sonic_policy_to_hardware()[7]) * 0.01f);
  expect_near("sonic planner resampled velocity", resampled[3].joint_vel[0], 30.f, 1e-4f);
  expect_throw("sonic planner rejects short prediction", [&] {
    (void)planner.resample_output(qpos, 23);
  });

  auto old_motion = resampled;
  auto new_motion = resampled;
  for (auto& frame : new_motion) {
    for (float& q : frame.joint_pos) q += 1.f;
  }
  planner.merge_generated(old_motion, 0);
  planner.merge_generated(new_motion, 2);
  expect_near("sonic planner blend starts from old motion", planner.motion()[2].joint_pos[0],
              old_motion[2].joint_pos[0]);
  expect_near("sonic planner blend reaches aligned new motion",
              planner.motion()[12].joint_pos[0], new_motion[10].joint_pos[0]);
}

}  // namespace

int main() {
  try {
    test_activation_blend();
    test_async_command_publisher();
    test_policy_contract();
    test_policy_protocol_constraints();
    test_command_safety_filter();
    test_pd_torque_filter();
    test_prearm_command();
    test_npy_reader();
    test_network_command_authentication();
    test_reference_endpoint_boundary();
    test_onnx_runner();
    test_realtime_input_stale_gate();
    test_sonic_policy_helpers();
    test_supermimic_policy_helpers();
    test_allmimic_policy_helpers();
    test_sonic_planner_helpers();
  } catch (const std::exception& e) {
    fail("uncaught exception", e.what());
  }

  if (g_failures != 0) {
    std::cerr << "[runtime_hardening_test] failures=" << g_failures << "\n";
    return 1;
  }
  std::cout << "[runtime_hardening_test] OK\n";
  return 0;
}
