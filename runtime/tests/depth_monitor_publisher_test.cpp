#include "action_go/runtime/depth_monitor_publisher.h"

#include <cassert>
#include <thread>

int main() {
  const std::string path = "/tmp/action_go_depth_monitor_test_" + std::to_string(getpid()) + ".sock";
  const std::vector<float> depth(58 * 87, 0.125f);
  const std::array<float, 4> quaternion{1.f, 0.f, 0.f, 0.f};
  action_go::runtime::DepthMonitorPublisher absent(path);
  absent.publish(depth, quaternion, true);
  const int receiver = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  assert(receiver >= 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  assert(bind(receiver, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  action_go::runtime::DepthMonitorPublisher publisher(path);
  publisher.publish(depth, quaternion, true);
  std::array<unsigned char, 36 + 58 * 87 * sizeof(float)> packet{};
  assert(recv(receiver, packet.data(), packet.size(), 0) == static_cast<ssize_t>(packet.size()));
  assert(std::memcmp(packet.data(), "AGDEPTH1", 8) == 0);
  assert(std::memcmp(packet.data() + 20, quaternion.data(), 4 * sizeof(float)) == 0);
  assert(std::memcmp(packet.data() + 36, depth.data(), depth.size() * sizeof(float)) == 0);
  publisher.publish(depth, quaternion, true);
  assert(recv(receiver, packet.data(), packet.size(), 0) < 0);
  for (int iteration = 0; iteration < 20; ++iteration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(105));
    const auto started = std::chrono::steady_clock::now();
    publisher.publish(depth, quaternion, true);
    assert(std::chrono::steady_clock::now() - started < std::chrono::milliseconds(100));
  }
  close(receiver);
  unlink(path.c_str());
  return 0;
}
