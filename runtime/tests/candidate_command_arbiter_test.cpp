#include "action_go/runtime/candidate_command_arbiter.h"
#include "action_go/runtime/event_logger.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

using action_go::runtime::ArbiterRobotState;
using action_go::runtime::CandidateCommandArbiter;
using action_go::runtime::CandidateCommandArbiterConfig;
using action_go::runtime::CommandArbiterState;
using action_go::runtime::CommandSelection;
using action_go::runtime::JointCommandCandidate;
using action_go::runtime::HandoffDecision;
using action_go::runtime::PolicyEntryContractV1;
using action_go::runtime::PreviewCapability;
using action_go::runtime::SourceExitStatusV1;

namespace {

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

CandidateCommandArbiter::TimePoint at_ms(int milliseconds) {
  return CandidateCommandArbiter::TimePoint{} + std::chrono::milliseconds(milliseconds);
}

ArbiterRobotState state_at(int milliseconds, float q = 0.f) {
  ArbiterRobotState state;
  state.q.assign(3, q);
  state.dq.assign(3, 0.f);
  state.connected = true;
  state.sequence = static_cast<uint64_t>(milliseconds);
  state.mode_machine = 5;
  state.stamp = at_ms(milliseconds);
  return state;
}

JointCommandCandidate command_at(int milliseconds, float q, uint64_t sequence) {
  JointCommandCandidate command;
  command.q.assign(3, q);
  command.dq.assign(3, 0.f);
  command.tau.assign(3, 0.f);
  command.kp.assign(3, 40.f);
  command.kd.assign(3, 2.f);
  command.mode_machine = 5;
  command.sequence = sequence;
  command.stamp = at_ms(milliseconds);
  return command;
}

CandidateCommandArbiter make_arbiter(float reject_jump = 0.f) {
  CandidateCommandArbiterConfig config;
  config.motor_count = 3;
  config.candidate_timeout_sec = 0.05;
  config.lowstate_timeout_sec = 0.10;
  config.reject_target_position_jump_rad = reject_jump;
  config.lower_limits = {-1.f, -1.f, -1.f};
  config.upper_limits = {1.f, 1.f, 1.f};
  return CandidateCommandArbiter(config);
}

SourceExitStatusV1 safe_exit_status() {
  SourceExitStatusV1 status;
  status.valid = true;
  status.sequence = 1;
  status.freshness_sec = 0.0;
  status.phase = "double_support";
  status.contact_mode = "double_support";
  status.contact_count = 2;
  status.base_height_m = 0.76;
  status.tilt_deg = 4.0;
  status.angular_speed_rps = 0.2;
  return status;
}

void test_source_preview_commit() {
  auto arbiter = make_arbiter();
  arbiter.set_context("run", "trial", "source__target", "source", "target");
  arbiter.update_lowstate(state_at(100));
  arbiter.update_source(command_at(100, 0.1f, 1));

  auto output = arbiter.tick(at_ms(101));
  expect(output.publish && output.selection == CommandSelection::kSource,
         "source must be selected initially");

  arbiter.request_preview();
  output = arbiter.tick(at_ms(102));
  expect(output.state == CommandArbiterState::kTargetPreview,
         "preview without target must remain in target_preview");
  expect(output.selection == CommandSelection::kSource,
         "source must continue during preview");

  arbiter.update_target(command_at(103, 0.2f, 9));
  output = arbiter.tick(at_ms(104));
  expect(output.state == CommandArbiterState::kTargetArmed,
         "fresh target must arm");
  expect(output.selection == CommandSelection::kSource,
         "arming must not transfer command ownership");
  expect(output.assessment.authorized, "armed target assessment must authorize");

  arbiter.request_commit();
  output = arbiter.tick(at_ms(105));
  expect(output.state == CommandArbiterState::kCommit,
         "commit request must enter commit");
  expect(output.selection == CommandSelection::kTarget,
         "commit must select target");
  expect(std::abs(output.command.q[0] - 0.2f) < 1e-6f,
         "target command must be forwarded unchanged inside limits");
}

void test_discard_and_stale_fallback() {
  auto arbiter = make_arbiter();
  arbiter.update_lowstate(state_at(200));
  arbiter.update_source(command_at(200, 0.1f, 1));
  arbiter.update_target(command_at(200, 0.2f, 2));
  arbiter.request_preview();
  auto output = arbiter.tick(at_ms(201));
  expect(output.state == CommandArbiterState::kTargetArmed, "target must arm");
  arbiter.request_discard();
  output = arbiter.tick(at_ms(202));
  expect(output.state == CommandArbiterState::kSourceActive,
         "discard must restore source state");
  expect(output.selection == CommandSelection::kSource, "discard must retain source");

  arbiter.update_target(command_at(203, 0.2f, 3));
  arbiter.request_preview();
  output = arbiter.tick(at_ms(204));
  arbiter.request_commit();
  output = arbiter.tick(at_ms(205));
  expect(output.selection == CommandSelection::kTarget, "target must commit before stale test");

  arbiter.update_lowstate(state_at(260));
  arbiter.update_source(command_at(260, 0.15f, 4));
  output = arbiter.tick(at_ms(261));
  expect(output.fallback, "stale committed target must trigger fallback");
  expect(output.state == CommandArbiterState::kHoldRecovery,
         "stale target must enter explicit hold/recovery state");
  expect(output.selection == CommandSelection::kHold,
         "post-commit failure must not blindly restore a source candidate");
  expect(output.assessment.decision == HandoffDecision::kHold,
         "post-commit failure must be labeled as hold");
}

void test_invalid_target_and_hold() {
  auto arbiter = make_arbiter();
  arbiter.update_lowstate(state_at(300, 0.3f));
  auto bad = command_at(300, 0.2f, 1);
  bad.q[1] = std::numeric_limits<float>::quiet_NaN();
  arbiter.update_target(bad);
  arbiter.request_preview();
  auto output = arbiter.tick(at_ms(301));
  expect(output.state == CommandArbiterState::kTargetPreview,
         "invalid target must not arm");
  expect(!output.assessment.authorized, "invalid target must be rejected");
  expect(output.selection == CommandSelection::kHold,
         "missing source must produce hold");
  expect(std::abs(output.command.q[0] - 0.3f) < 1e-6f,
         "hold must use current pose");
  expect(output.command.mode_machine == 5,
         "hold must preserve the current controller mode_machine");
}

void test_jump_rejection_and_limit_filter() {
  auto arbiter = make_arbiter(0.25f);
  arbiter.update_lowstate(state_at(400));
  arbiter.update_source(command_at(400, 0.f, 1));
  auto target = command_at(400, 2.f, 2);
  arbiter.update_target(target);
  arbiter.request_preview();
  auto output = arbiter.tick(at_ms(401));
  expect(output.state == CommandArbiterState::kTargetPreview,
         "large jump must remain in preview");
  expect(!output.assessment.authorized, "large jump must reject target");
  expect(output.assessment.max_position_jump_rad >= 0.99,
         "jump metric must observe filtered limit boundary");
}

void test_lowstate_stale_stops_publication() {
  auto arbiter = make_arbiter();
  arbiter.update_lowstate(state_at(500));
  arbiter.update_source(command_at(610, 0.f, 1));
  auto output = arbiter.tick(at_ms(611));
  expect(!output.publish, "stale LowState must prevent normal publication");
  expect(output.reason.find("lowstate stale") != std::string::npos,
         "stale LowState reason must be explicit");
}

void test_entry_contract_validation() {
  PolicyEntryContractV1 contract;
  contract.policy = "target";
  contract.preview_capability = PreviewCapability::kTransactional;
  contract.entry_feature_names = {"q0", "dq0"};
  contract.entry_center = {0.0, 0.0};
  contract.entry_scale = {0.1, 0.2};
  std::string error;
  expect(action_go::runtime::validate_policy_entry_contract(contract, error),
         "valid entry contract must pass");
  contract.entry_scale.pop_back();
  expect(!action_go::runtime::validate_policy_entry_contract(contract, error),
         "mismatched entry vectors must fail");
  contract.entry_scale = {0.1, 0.2};
  contract.native_startup_mode = "disabled_by_preview";
  expect(!action_go::runtime::validate_policy_entry_contract(contract, error),
         "invalid native startup semantics must fail");
}

void test_source_exit_gate_rejects_airborne_and_retains_source() {
  CandidateCommandArbiterConfig config;
  config.motor_count = 3;
  config.candidate_timeout_sec = 0.05;
  config.lowstate_timeout_sec = 0.10;
  config.source_exit_gate.required = true;
  auto arbiter = CandidateCommandArbiter(config);
  arbiter.update_lowstate(state_at(700));
  arbiter.update_source(command_at(700, 0.1f, 1));
  arbiter.update_target(command_at(700, 0.2f, 2));
  auto status = safe_exit_status();
  status.phase = "back_hand_flip_airborne";
  status.airborne = true;
  status.contact_count = 0;
  arbiter.update_source_exit_status(status);
  arbiter.request_preview();
  auto output = arbiter.tick(at_ms(701));
  expect(output.state == CommandArbiterState::kSourceActive,
         "airborne source must not leave source_active");
  expect(output.selection == CommandSelection::kSource,
         "airborne rejection must retain the source command");
  expect(output.assessment.decision == HandoffDecision::kRejectUnsafeSource,
         "airborne rejection must be explicit in the assessment");
  expect(!output.assessment.source_exit_admissible,
         "airborne source must fail the exit gate");
}

void test_source_exit_gate_allows_supported_phase() {
  CandidateCommandArbiterConfig config;
  config.motor_count = 3;
  config.candidate_timeout_sec = 0.05;
  config.lowstate_timeout_sec = 0.10;
  config.source_exit_gate.required = true;
  auto arbiter = CandidateCommandArbiter(config);
  arbiter.update_lowstate(state_at(800));
  arbiter.update_source(command_at(800, 0.1f, 1));
  arbiter.update_target(command_at(800, 0.2f, 2));
  arbiter.update_source_exit_status(safe_exit_status());
  arbiter.request_preview();
  auto output = arbiter.tick(at_ms(801));
  expect(output.state == CommandArbiterState::kTargetArmed,
         "supported source phase must allow target arming");
  arbiter.request_commit();
  output = arbiter.tick(at_ms(802));
  expect(output.selection == CommandSelection::kTarget,
         "supported source phase must allow direct commit");
  expect(output.assessment.decision == HandoffDecision::kDirect,
         "successful commit must report direct decision");
}

void test_source_exit_gate_routes_fallen_state_to_recovery() {
  CandidateCommandArbiterConfig config;
  config.motor_count = 3;
  config.candidate_timeout_sec = 0.05;
  config.lowstate_timeout_sec = 0.10;
  config.source_exit_gate.required = true;
  auto arbiter = CandidateCommandArbiter(config);
  arbiter.update_lowstate(state_at(900));
  arbiter.update_source(command_at(900, 0.1f, 1));
  auto status = safe_exit_status();
  status.fallen = true;
  status.phase = "fallen";
  arbiter.update_source_exit_status(status);
  arbiter.request_preview();
  const auto output = arbiter.tick(at_ms(901));
  expect(output.state == CommandArbiterState::kSourceActive,
         "fallen source must not commit a normal target");
  expect(output.assessment.decision == HandoffDecision::kRecover,
         "fallen source must request dedicated recovery");
}

void test_source_exit_gate_rejects_stale_status() {
  CandidateCommandArbiterConfig config;
  config.motor_count = 3;
  config.candidate_timeout_sec = 0.05;
  config.lowstate_timeout_sec = 0.10;
  config.source_exit_gate.required = true;
  auto arbiter = CandidateCommandArbiter(config);
  arbiter.update_lowstate(state_at(950));
  arbiter.update_source(command_at(950, 0.1f, 1));
  auto status = safe_exit_status();
  status.freshness_sec = 1.0;
  arbiter.update_source_exit_status(status);
  arbiter.request_preview();
  const auto output = arbiter.tick(at_ms(951));
  expect(output.state == CommandArbiterState::kSourceActive,
         "stale source-exit status must reject preview");
  expect(output.reason.find("stale") != std::string::npos,
         "stale source-exit rejection must be explicit");
}

void test_hrm_authorization_and_rule_only_nan() {
  auto rule_only = make_arbiter();
  rule_only.update_lowstate(state_at(1000));
  rule_only.update_source(command_at(1000, 0.1f, 1));
  rule_only.update_target(command_at(1000, 0.2f, 2));
  rule_only.request_preview();
  auto output = rule_only.tick(at_ms(1001));
  expect(output.state == CommandArbiterState::kTargetArmed,
         "rule-only mode must still arm a valid target");
  expect(std::isnan(output.assessment.failure_probability),
         "rule-only mode must not invent a risk probability");

  auto hrm = make_arbiter();
  hrm.set_hrm_risk_evaluator(
      [](const action_go::runtime::HrmFeatureVectorV1& features)
          -> std::optional<double> {
        expect(features.values.size() == 16,
               "HRM feature contract must contain 16 values");
        return 0.8;
      },
      0.4);
  hrm.update_lowstate(state_at(1010));
  hrm.update_source(command_at(1010, 0.1f, 1));
  hrm.update_target(command_at(1010, 0.2f, 2));
  hrm.request_preview();
  output = hrm.tick(at_ms(1011));
  expect(output.state == CommandArbiterState::kTargetPreview,
         "HRM risk above threshold must keep the target in preview");
  expect(std::abs(output.assessment.failure_probability - 0.8) < 1e-9,
         "HRM probability must be reported without fabrication");
}

void test_nonfinite_event_numbers_are_valid_json_null() {
  const auto path = std::filesystem::temp_directory_path() /
      ("action_go_event_logger_test_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".jsonl");
  action_go::runtime::RuntimeEventLogger logger;
  logger.open(path.string());
  logger.log("nonfinite",
             {action_go::runtime::event_number(
                  "positive_inf", std::numeric_limits<double>::infinity()),
              action_go::runtime::event_number(
                  "not_a_number", std::numeric_limits<double>::quiet_NaN())});
  std::ifstream input(path);
  const std::string line((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  std::error_code error;
  std::filesystem::remove(path, error);
  expect(line.find("\"positive_inf\":null") != std::string::npos,
         "positive infinity must serialize as JSON null");
  expect(line.find("\"not_a_number\":null") != std::string::npos,
         "NaN must serialize as JSON null");
}

}  // namespace

int main() {
  try {
    test_source_preview_commit();
    test_discard_and_stale_fallback();
    test_invalid_target_and_hold();
    test_jump_rejection_and_limit_filter();
    test_lowstate_stale_stops_publication();
    test_entry_contract_validation();
    test_source_exit_gate_rejects_airborne_and_retains_source();
    test_source_exit_gate_allows_supported_phase();
    test_source_exit_gate_routes_fallen_state_to_recovery();
    test_source_exit_gate_rejects_stale_status();
    test_hrm_authorization_and_rule_only_nan();
    test_nonfinite_event_numbers_are_valid_json_null();
    std::cout << "candidate_command_arbiter_test: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "candidate_command_arbiter_test: FAIL: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
