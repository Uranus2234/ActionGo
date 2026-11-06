#pragma once

#include <memory>
#include <string>

namespace action_go::runtime {

class PolicyCommandRelay {
 public:
  virtual ~PolicyCommandRelay() = default;
  virtual void add(const std::string& topic) = 0;
  virtual bool ready(const std::string& topic) = 0;
  virtual bool select(const std::string& topic) = 0;
  virtual void release(const std::string& topic) = 0;
};

std::shared_ptr<PolicyCommandRelay> make_policy_command_relay(
    const std::string& output_topic = "rt/user_lowcmd",
    const std::string& state_topic = "rt/lowstate");

}
