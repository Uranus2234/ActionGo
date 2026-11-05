#include "action_go/runtime/bfm_action.h"
#include "action_go/runtime/bfm_observation.h"
#include "action_go/runtime/pbfm_policy.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace action_go::runtime;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void compare(const std::vector<float>& actual, const nlohmann::json& expected,
             float tolerance, const std::string& name) {
  const auto reference = expected.get<std::vector<float>>();
  require(actual.size() == reference.size(), name + " dimension mismatch");
  for (size_t index = 0; index < actual.size(); ++index) {
    require(std::isfinite(actual[index]) &&
                std::abs(actual[index] - reference[index]) <= tolerance * (1.f + std::abs(reference[index])),
            name + " mismatch at index " + std::to_string(index));
  }
}

void test_history_timing() {
  BfmObservationHistory previous;
  BfmObservationHistory current;
  previous.reset(1, 4);
  current.reset(1, 4);
  for (int step = 1; step <= 8; ++step) {
    const std::vector<float> position{static_cast<float>(step)};
    const auto prior = previous.build(position, {0.f}, {0.f}, {0.f, 0.f, -1.f},
                                       {0.f, 0.f, 0.f}, position, BfmHistoryTiming::kPreviousOnly);
    const auto inclusive = current.build(position, {0.f}, {0.f}, {0.f, 0.f, -1.f},
                                         {0.f, 0.f, 0.f}, position, BfmHistoryTiming::kIncludeCurrent);
    for (size_t age = 0; age < 4; ++age) {
      require(prior.history[age] == std::max(0, step - 1 - static_cast<int>(age)),
              "previous_only history is off by one");
      require(inclusive.history[age] == std::max(0, step - static_cast<int>(age)),
              "include_current history changed");
    }
  }
  const auto decoded = decode_bfm_action({-2.f, 0.5f, 2.f}, 5.f,
      {0.f, 1.f, 0.f}, {1.f, 0.1f, 1.f}, {-2.f, -2.f, -2.f}, {2.f, 2.f, 2.f});
  require(decoded.scaled_action == std::vector<float>{-5.f, 2.5f, 5.f}, "scaled action mismatch");
  require(decoded.target_position == std::vector<float>{-2.f, 1.25f, 2.f}, "target clipping mismatch");
  bool rejected = false;
  try {
    decode_bfm_action({std::numeric_limits<float>::quiet_NaN()}, 5.f, {0.f}, {1.f}, {-1.f}, {1.f});
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "nonfinite action was accepted");
  BfmContext context;
  context.heading_valid = true;
  const auto heading = context.heading_input(1.f, 0.5f);
  require(std::abs(heading[1] - std::sin(0.5f)) < 1e-6f, "heading left/right sign mismatch");
  context.heading_valid = false;
  require(context.heading_input(1.f, 0.5f) == std::array<float, 2>{0.f, 0.f}, "invalid heading mismatch");
  context.velocity_command = {1.f, 0.f, 1.f};
  BfmContextSequence sequence;
  sequence.frames.push_back(context);
  const auto rotated = sequence.at(0, 0.f, static_cast<float>(M_PI / 2));
  require(std::abs(rotated.velocity_command[0]) < 1e-6f &&
              std::abs(rotated.velocity_command[1] + 1.f) < 1e-6f,
          "tracking velocity was not transformed from initial yaw to current yaw");
  bool exhausted = false;
  try {
    sequence.at(1, 0.f, 0.f);
  } catch (const std::out_of_range&) {
    exhausted = true;
  }
  require(exhausted, "tracking sequence silently reused the final frame");
}

void test_depth_history() {
  PbfmDepthHistory history;
  for (int step = 1; step <= 50; ++step) {
    const auto stack = history.append(std::vector<float>(36 * 64, step / 255.f * 2.5f));
    require(stack.size() == 8 * 18 * 32, "depth history size mismatch");
    size_t frame = 0;
    for (int age : {35, 30, 25, 20, 15, 10, 5, 0}) {
      const float expected = std::max(1, step - age) / 255.f;
      for (size_t pixel = 0; pixel < 18 * 32; ++pixel)
        require(std::abs(stack[frame * 18 * 32 + pixel] - expected) < 1e-7f, "depth history age mismatch");
      ++frame;
    }
  }
}

void test_directional_contexts() {
  PbfmSpec spec;
  spec.idle_context.latent = {1.f, 1.f};
  for (const auto& entry : std::vector<std::pair<std::string, std::array<float, 2>>>{
           {"forward", {0.6f, 0.f}}, {"backward", {-0.3f, 0.f}},
           {"left", {0.f, 0.4f}}, {"right", {0.f, -0.4f}}}) {
    BfmContext context;
    context.velocity_command = {entry.second[0], entry.second[1], 1.f};
    context.latent = {entry.second[0] == 0.f ? 0.f : std::copysign(16.f, entry.second[0]),
                      entry.second[1] == 0.f ? 0.f : std::copysign(16.f, entry.second[1])};
    spec.directional_contexts.emplace(entry.first, context);
    require(spec.movement_context(entry.second[0], entry.second[1]).latent == context.latent,
            "remote direction selected the wrong latent");
  }
  require(spec.movement_context(0.f, 0.f).latent == spec.idle_context.latent,
          "turn-only command must use the idle latent");
  const auto diagonal = spec.movement_context(0.6f, 0.4f);
  require(std::abs(diagonal.latent[0] - std::sqrt(128.f)) < 1e-5f &&
              std::abs(diagonal.latent[1] - std::sqrt(128.f)) < 1e-5f,
          "diagonal direction did not preserve latent normalization");
  BfmContextSequence cycle;
  cycle.frames.push_back(spec.directional_contexts.at("forward"));
  cycle.frames.push_back(spec.directional_contexts.at("forward"));
  cycle.frames[1].latent = {0.f, 16.f};
  spec.directional_sequences.emplace("forward", cycle);
  require(spec.movement_context(1.2f, 0.f, 1).latent == cycle.frames[1].latent,
          "walking cycle did not advance");
  require(spec.movement_context(1.2f, 0.f, 2).latent == cycle.frames[0].latent,
          "walking cycle did not loop");
  require(spec.movement_context(0.f, 0.f, 1).latent == spec.idle_context.latent,
          "release retained walking cycle instead of idle");
}

void test_joystick_contexts() {
  PbfmSpec spec;
  spec.proportional_speed = true;
  spec.max_forward_speed = 1.f;
  spec.max_backward_speed = 0.3f;
  spec.max_lateral_speed = 0.4f;
  spec.max_yaw_rate = 0.6f;
  spec.idle_context.latent = {16.f, 0.f};
  BfmContext forward;
  forward.latent = {0.f, 16.f};
  forward.velocity_command = {1.f, 0.f, 1.f};
  spec.directional_contexts.emplace("forward", forward);
  float previous_angle = -1.f;
  for (float amplitude : {0.f, 0.25f, 0.5f, 0.75f, 1.f}) {
    const auto context = spec.movement_context(amplitude, 0.f);
    require(std::abs(std::hypot(context.latent[0], context.latent[1]) - 16.f) < 1e-5f,
            "joystick interpolation changed latent norm");
    const float angle = std::atan2(context.latent[1], context.latent[0]);
    require(angle > previous_angle, "joystick amplitude did not continuously increase gait strength");
    previous_angle = angle;
  }
  BfmContextSequence left;
  left.frames = {forward, spec.idle_context};
  BfmContextSequence right;
  auto right_frame = forward;
  right_frame.latent = {16.f, 0.f};
  right.frames = {right_frame};
  spec.turn_sequences.emplace("left", left);
  spec.turn_sequences.emplace("right", right);
  require(spec.turning_context(0.6f, 2).latent == forward.latent, "left turn did not loop its stepping context");
  require(spec.turning_context(-0.6f, 0).latent == right_frame.latent, "right turn selected the wrong context");
  const auto small_turn = spec.turning_context(0.15f, 0);
  require(small_turn.latent[1] > small_turn.latent[0] && small_turn.latent[0] > 0.f,
          "small turn did not interpolate left/right stepping contexts");
  require(std::abs(std::hypot(small_turn.latent[0], small_turn.latent[1]) - 16.f) < 1e-5f,
          "small turn changed latent normalization");
  require(spec.turning_context(0.f, 1).latent == spec.idle_context.latent, "turn release did not select idle");
}

void test_joystick_heading() {
  PbfmHeadingController steering;
  steering.reset(0.f);
  for (int step = 0; step < 300; ++step) {
    const float target = steering.update(1.f, 0.6f, 0.25f, 0.f, 0.02f);
    require(target > 0.f && target <= 0.250001f, "steering accumulated an unreachable heading target");
  }
  require(std::abs(steering.update(0.f, 0.6f, 0.25f, 0.12f, 0.02f) - 0.12f) < 1e-6f,
          "turn release kept chasing the old heading");
  require(std::abs(steering.update(0.f, 0.6f, 0.25f, 0.17f, 0.02f) - 0.12f) < 1e-6f,
          "straight walking reset the held heading");
  steering.reset(3.13f);
  const float wrapped = steering.update(1.f, 0.6f, 0.25f, 3.13f, 0.02f);
  require(std::abs(std::remainder(wrapped - 3.13f, 2.f * static_cast<float>(M_PI)) - 0.012f) < 1e-5f,
          "steering failed across the yaw wrap");
  steering.reset(0.f);
  steering.update(1.f, 1.f, 0.f, 0.f, 1.f);
  require(steering.update(0.f, 1.f, 0.f, 0.2f, 0.02f) == 1.f,
          "legacy heading integration changed without the steering opt-in");
}

void test_replay(const std::string& fixture) {
  std::ifstream stream(fixture);
  nlohmann::json cases;
  stream >> cases;
  for (const auto& scenario : cases) {
    OnnxRunner runner;
    auto model_path = std::filesystem::path(scenario.at("model").get<std::string>());
    if (model_path.is_relative()) model_path = std::filesystem::path(fixture).parent_path() / model_path;
    runner.load(model_path.string(), 2);
    for (const auto& sample : scenario.value("actor_cases", nlohmann::json::array())) {
      const auto action = runner.run_single(sample.at("actor_obs").get<std::vector<float>>(),
                                             "actor_obs", "action", 29);
      compare(action, sample.at("action"), 2e-4f, "exported actor parity");
    }
    const bool pbfm = scenario.at("variant") == "pbfm";
    const bool previous_only = scenario.value("history_timing", pbfm ? "previous_only" : "include_current") == "previous_only";
    if (pbfm) PbfmPolicy::validate_model(runner);
    BfmObservationHistory history;
    history.reset(29, 4);
    std::vector<float> last_action(29, 0.f);
    const auto defaults = scenario.at("default_angles").get<std::vector<float>>();
    BfmContext context;
    context.latent = scenario.at("z").get<std::vector<float>>();
    context.heading_relative = scenario.at("heading_relative").get<std::array<float, 2>>();
    context.heading_valid = scenario.at("heading_valid").get<bool>();
    context.velocity_command = scenario.at("velocity_command").get<std::array<float, 3>>();
    for (const auto& frame : scenario.at("frames")) {
      const auto observation = history.build(frame.at("q"), frame.at("dq"), defaults,
          frame.at("gravity").get<std::array<float, 3>>(), frame.at("gyro").get<std::array<float, 3>>(),
          last_action, previous_only ? BfmHistoryTiming::kPreviousOnly : BfmHistoryTiming::kIncludeCurrent);
      compare(observation.state, frame.at("state"), 1e-6f, "state");
      compare(observation.last_action, frame.at("last_action"), 2e-4f, "last action");
      compare(observation.history, frame.at("history_actor"), 2e-4f, "history");
      std::vector<float> action;
      if (pbfm) {
        const auto heading = context.heading_input(0.f, frame.at("yaw").get<float>());
        compare({heading.begin(), heading.end()}, frame.at("heading"), 1e-6f, "heading");
        const auto output = PbfmPolicy::infer(runner, observation, context, heading,
                                             frame.at("depth").get<std::vector<float>>());
        action = output.action;
        compare(output.estimated_velocity, frame.at("estimated_velocity"), 2e-4f, "estimated velocity");
      } else {
        const auto actor_input = observation.actor_input(context.latent);
        compare(actor_input, frame.at("actor_obs"), 2e-4f, "legacy actor input");
        action = runner.run_single(actor_input, "actor_obs", "action", 29);
      }
      compare(action, frame.at("action"), 2e-4f, "raw action");
      auto decoded = decode_bfm_action(action, 5.f, defaults, scenario.at("action_scale"),
                                        scenario.at("lower"), scenario.at("upper"));
      compare(decoded.scaled_action, frame.at("scaled_action"), 2e-4f, "scaled action");
      compare(decoded.target_position, frame.at("target_q"), 2e-4f, "target q");
      last_action = std::move(decoded.scaled_action);
    }
    std::cout << "replay passed: " << scenario.at("name") << " frames=" << scenario.at("frames").size() << '\n';
  }
}

}

int main(int argc, char** argv) {
  try {
    test_history_timing();
    test_depth_history();
    test_directional_contexts();
    test_joystick_contexts();
    test_joystick_heading();
    if (argc == 2) test_replay(argv[1]);
    std::cout << "BFM policy checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
