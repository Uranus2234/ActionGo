#include "action_go/runtime/command_timing.h"
#include "action_go/runtime/runtime_session.h"
#include "action_go/runtime/async_command_publisher.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <thread>

using namespace action_go::runtime;

int main() {
  std::string pattern = (std::filesystem::temp_directory_path() / "actiongo_logging_XXXXXX").string();
  assert(mkdtemp(pattern.data()));
  const std::filesystem::path root(pattern);
  const auto path = (root / "events.jsonl").string();
  {
    RuntimeEventLogger logger;
    logger.open(path);
    logger.log("escaped", {event_string("value", std::string("line\n\t\001"))});
    logger.flush();
    std::ifstream input(path);
    const auto event = nlohmann::json::parse(input);
    assert(event.at("value") == std::string("line\n\t\001"));
    assert(event.at("schema_version") == 1);
    assert(event.at("level") == "INFO");
    logger.log("control_failure", {event_bool("ok", false)});
  }
  {
    std::ifstream input(path);
    std::string line;
    int count = 0;
    while (std::getline(input, line)) {
      const auto parsed = nlohmann::json::parse(line);
      assert(parsed.is_object());
      ++count;
    }
    assert(count == 2);
  }
  {
    RuntimeEventLogger logger;
    logger.open_append(path);
    logger.log("appended");
    logger.flush();
  }
  setenv("ACTION_GO_LOG_MAX_BYTES", "4096", 1);
  {
    RuntimeEventLogger logger;
    logger.open((root / "rotating.jsonl").string());
    for (int index = 0; index < 6; ++index) {
      logger.log("large", {event_string("data", std::string(5000, 'x'))});
      logger.flush();
    }
  }
  assert(std::filesystem::exists(root / "rotating.jsonl.3"));
  assert(!std::filesystem::exists(root / "rotating.jsonl.4"));
  assert(std::filesystem::exists(root / "rotating.log.3"));
  assert(!std::filesystem::exists(root / "rotating.jsonl.log"));
  {
    std::ifstream input(root / "rotating.log");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    assert(text.find("large") != std::string::npos);
  }
  unsetenv("ACTION_GO_LOG_MAX_BYTES");
  {
    RuntimeEventLogger logger;
    std::filesystem::create_symlink("/dev/full", root / "failed.jsonl");
    logger.open((root / "failed.jsonl").string());
    logger.log("write_failure_fixture");
    logger.flush();
    for (int attempt = 0; attempt < 100 && logger.healthy(); ++attempt)
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(!logger.healthy());
    logger.flush();
  }
  {
    std::ifstream input(root / "failed.jsonl.health.json");
    assert(nlohmann::json::parse(input).at("failed") == true);
  }
  {
    RuntimeEventLogger logger;
    const auto timing_path = (root / "timing.jsonl").string();
    logger.open(timing_path);
    CommandTimingReporter reporter;
    const auto received = std::chrono::steady_clock::now() - std::chrono::milliseconds(10);
    {
      CommandTimingScope timing(received, 42);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const auto provenance = finish_command_timing();
    JointCommand command{{0.f}, {1.f}, {1.f}};
    {
      AsyncCommandPublisher publisher(200, 0.0, command,
          [&](const JointCommand& published, bool active) {
            if (active) reporter.published(logger, published.timing);
          });
      publisher.update(command);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      publisher.stop_and_join();
      publisher.rethrow_if_failed();
    }
    logger.flush();
    std::ifstream input(timing_path);
    std::string line;
    assert(std::getline(input, line));
    const auto timing = nlohmann::json::parse(line);
    assert(timing.at("state_seq") == 42);
    assert(timing.at("command_seq") == provenance.command_sequence);
    assert(timing.at("e2e_ms").get<double>() >= 12);
    assert(timing.at("inference_ms").get<double>() >= 2);
    assert(!std::getline(input, line));
    assert(std::filesystem::exists(timing_path + ".latest.json"));
  }
  setenv("ACTION_GO_LOG_DIR", root.c_str(), 1);
  {
    RuntimeSession session("", "", "", "lo", true);
    std::cout << "captured diagnostic\n" << std::flush;
    std::cerr << "captured error\n" << std::flush;
    session.render("WALKRUN", "", "");
    std::this_thread::sleep_for(std::chrono::milliseconds(205));
    session.render("USER_CTRL", "test_policy", (root / "timing.jsonl").string());
    std::this_thread::sleep_for(std::chrono::milliseconds(205));
    session.render("DAMPING", "", "");
    session.notice("INFO", "Test complete");
  }
  bool captured = false;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_directory()) continue;
    std::ifstream input(entry.path() / "runtime.output.log");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    captured = text.find("captured diagnostic") != std::string::npos &&
               text.find("captured error") != std::string::npos;
    assert(std::filesystem::exists(entry.path() / "summary.json"));
  }
  assert(captured);
  unsetenv("ACTION_GO_LOG_DIR");
  std::filesystem::remove_all(root);
}
