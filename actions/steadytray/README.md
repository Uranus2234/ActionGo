# SteadyTray action_go package

Runs the SteadyTray student policy through `action_go`.

Object observation sources:

- `apriltag`: RealSense color + AprilTag pose inside `generic_sensor_policy_player` (requires librealsense2 + AprilRobotics apriltag C library at build time).
- `udp`: receive one pose packet `x y z qw qx qy qz` on `--udp-port` (text or binary 7x float32), useful if AprilTag pose is produced by a separate Python node.
- `zero`: identity object pose for model/DDS smoke tests only.

Build-time dependencies for in-process AprilTag:

```bash
# librealsense2 must come from the non-ROS install used by action_go.
# AprilTag C library (AprilRobotics/apriltag):
git clone https://github.com/AprilRobotics/apriltag.git
cmake -S apriltag -B apriltag/build -DCMAKE_BUILD_TYPE=Release
cmake --build apriltag/build -j
sudo cmake --install apriltag/build

cd ~/action_go
cmake -S . -B build -Drealsense2_DIR=/usr/local/lib/cmake/realsense2
cmake --build build --target generic_sensor_policy_player -j2
```

If CMake cannot find AprilTag, `generic_sensor_policy_player` still builds for `udp/zero`, but `--object-source apriltag` exits with an explanatory error.

Recommended test order:

```bash
# model/config only
build/generic_sensor_policy_player --spec actions/steadytray/config/generic_steadytray.yaml --check-only

# camera / tag only
build/generic_sensor_policy_player --spec actions/steadytray/config/generic_steadytray.yaml --object-source apriltag --tag-size 0.08 --tag-id 0 --require-tag --camera-check

# DDS + policy, no LowCmd publish
build/generic_sensor_policy_player --spec actions/steadytray/config/generic_steadytray.yaml --net enP8p1s0 --object-source zero \
  --dds-dry-run --duration-sec 5
```

For real runs, set the printed tag size in meters. If the AprilTag frame is not the same as the SteadyTray object frame, use:

```bash
--tag-to-object-pos "x y z" --tag-to-object-quat "qw qx qy qz"
```

If the AprilTag detector reports pose in a RealSense optical frame that differs from
the policy's `d435_link` frame, add the fixed transform:

```bash
--vision-to-policy-pos "x y z" --vision-to-policy-quat "qw qx qy qz"
```

The player computes:

```text
T_policy_object = T_policy_vision * T_vision_tag * T_tag_object
```
