# OmniXtreme action_go player

This package wires the OmniXtreme G1 ONNX policy into action_go.

Resources may be symlinked from an external checkout configured by
`OMNIXTREME_ROOT`:

```text
actions/omnixtreme/model/base_policy_trt.onnx -> $OMNIXTREME_ROOT/policy/base_policy_trt.onnx
actions/omnixtreme/model/residual_policy.onnx -> $OMNIXTREME_ROOT/policy/residual_policy.onnx
actions/omnixtreme/model/fk_trt.onnx -> $OMNIXTREME_ROOT/policy/fk_trt.onnx
actions/omnixtreme/motions/motion.npz -> $OMNIXTREME_ROOT/policy/motion.npz
```

`motion.npz` is a concatenated multi-segment motion.  The split/renamed
segments live under `actions/omnixtreme/motions/segments/` and are registered as
action_go actions in `actions/omnixtreme/omnixtreme_segments.yaml`.

## Check only

```bash
./build/generic_composite_policy_player \
  --spec actions/omnixtreme/config/generic_omnixtreme.yaml \
  --check-only
```

## action_go simulation interface

```bash
/usr/bin/python3 tools/sim/omnixtreme_unitree_mujoco_viewer.py --max-frames 1000
```

Default selected segment, currently segment 4 (`4_wbst_stand.npz`):

```bash
/usr/bin/python3 tools/sim/omnixtreme_unitree_mujoco_viewer.py \
  --action omni_motion_default
```

You can select any registered OmniXtreme segment directly:

```bash
/usr/bin/python3 tools/sim/omnixtreme_unitree_mujoco_viewer.py \
  --action omni_seg04_wbst_stand
```

The default simulation bridge is `--bridge native`: it uses OmniXtreme's original
`robots/g1/no_hand.xml`, `sim-dt=0.004`, and a lock-step DDS bridge.  ONNX
inference is done by `build/generic_composite_policy_player`, the same C++ player used by
action_go.  The older Unitree bridge can still be selected with
`--bridge unitree --sim-dt 0.001`, but it does not match `deploy_mujoco.py`
closely enough for the 500-1000 frame high-dynamic segment.

On the real robot, `generic_composite_policy_player` runs in `decoupled-publish` mode: ONNX
inference stays at 50 Hz while LowCmd publication continues at 500 Hz in a
separate thread.  This avoids command dropouts during inference.  The log prints
`infer_ms(...)`, `infer_late`, and `pub_late`; if `infer_late` increases on the
robot, the onboard CPU is not keeping the 50 Hz policy loop.

For real dynamic OmniXtreme segments, use ONNX Runtime with TensorRT EP.  The
paper reports about 10 ms end-to-end inference on the G1 Orin NX with TensorRT.
The bundled `thirdparty/onnxruntime-linux-aarch64-1.22.0` may be CPU-only; a
valid TensorRT build must provide at least:

```text
libonnxruntime.so
libonnxruntime_providers_shared.so
libonnxruntime_providers_cuda.so
libonnxruntime_providers_tensorrt.so
```

### Build policy: regular players vs OmniXtreme

`generic_composite_policy_player` is built by default and can run `--check-only`
with the bundled CPU ONNX Runtime.  For real dynamic OmniXtreme segments, the
recommended robot build is still a **split build**:

- `generic_policy_player` covers state/mimic policies,
  `generic_motion_tracking_player` covers motion-conditioned tracking policies,
  and `generic_sensor_policy_player` covers Instinct and SteadyTray. These
  non-Omni executables use the bundled regular ORT under `thirdparty/`;
- only `generic_composite_policy_player` is built against the TensorRT-enabled ORT.

First build the regular tree.  This also produces a CPU/auto
`generic_composite_policy_player` for configuration and asset checks:

```bash
cd ~/action_go
cmake -S . -B build \
  -Drealsense2_DIR=/usr/local/lib/cmake/realsense2 \
  -DACTION_GO_ONNXRUNTIME_DIR=$PWD/thirdparty/onnxruntime-linux-aarch64-1.22.0
cmake --build build --parallel $(nproc)
```

Then rebuild only OmniXtreme with the correct TensorRT ORT for your JetPack and
copy that binary over the CPU/auto one:

```bash
# JetPack 5.x / Ubuntu 20.04 example: ORT v1.14.1
export OMNI_ORT=$HOME/onnxruntime-trt-1.14.1

# JetPack 6.x / Ubuntu 22.04 example: ORT v1.19.2
# export OMNI_ORT=$HOME/onnxruntime-trt-1.19.2

cmake -S . -B build_omni_trt \
  -Drealsense2_DIR=/usr/local/lib/cmake/realsense2 \
  -DACTION_GO_BUILD_OMNIXTREME=ON \
  -DACTION_GO_ONNXRUNTIME_DIR=$OMNI_ORT
cmake --build build_omni_trt --target generic_composite_policy_player --parallel $(nproc)
cp -a build_omni_trt/generic_composite_policy_player build/generic_composite_policy_player
```

Do not globally prepend one ORT for this split build.  The launcher scripts use
`ACTION_GO_ORT_MODE=clean` by default: they remove ONNX Runtime entries from
`LD_LIBRARY_PATH` and let each executable use its own build RPATH.  This is safe
on the old Ubuntu 20.04 / JetPack 5.x machines too; no `ACTION_GO_ORT` export is
needed for the normal launch scripts.

Verify the split build with a clean environment:

```bash
env -u LD_LIBRARY_PATH ldd build/generic_policy_player | grep onnxruntime
env -u LD_LIBRARY_PATH ldd build/generic_composite_policy_player | grep onnxruntime
```

Expected:

```text
generic_policy_player / other non-Omni players -> ~/action_go/thirdparty/onnxruntime-linux-aarch64-1.22.0/...
generic_composite_policy_player -> ~/onnxruntime-trt-1.14.1/...   # JetPack 5.x / Ubuntu 20.04
generic_composite_policy_player -> ~/onnxruntime-trt-1.19.2/...   # JetPack 6.x / Ubuntu 22.04
```

Robot-side sanity check:

```bash
env -u LD_LIBRARY_PATH ./build/generic_composite_policy_player \
  --spec actions/omnixtreme/config/generic_omnixtreme.yaml \
  --trajectory actions/omnixtreme/motions/segments/4_wbst_stand.npz \
  --ep tensorrt --trt-fp16 --check-only
```

It should print `TensorrtExecutionProvider`.  If it prints only
`CPUExecutionProvider`, the Omni binary is still linked to CPU-only ONNX Runtime
or cannot see `libonnxruntime_providers_tensorrt.so`.

### Jetson TensorRT / ONNXRuntime setup

The G1 onboard Jetson used in our tests is a JetPack 5.x stack.  For Ubuntu
22.04 / JetPack 6.x, do **not** use the v1.14.1 recipe; use the JP6 recipe
below.

```text
L4T / JetPack: R35.3.1
CUDA: 11.4
cuDNN: 8.6
TensorRT: 8.5.2 + CUDA 11.4
ONNXRuntime: v1.14.1 built from source with CUDAExecutionProvider and TensorrtExecutionProvider
```

Check the robot-side stack:

```bash
cat /etc/nv_tegra_release
nvcc --version
dpkg -l | grep -E 'libnvinfer|libcudnn|cuda-toolkit' | head -30
find . /usr /opt "$HOME" -name 'libonnxruntime_providers_tensorrt.so' 2>/dev/null
```

If `libonnxruntime_providers_tensorrt.so` is not found, the current
ONNXRuntime is not a TensorRT-enabled build.

Build ONNXRuntime v1.14.1 with TensorRT EP on Jetson:

```bash
cd ~
git clone --recursive https://github.com/microsoft/onnxruntime.git onnxruntime-trt
cd ~/onnxruntime-trt
git checkout v1.14.1
git submodule update --init --recursive

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git build-essential cmake libopenblas-dev \
  python3-pip python3-dev python3-setuptools python3-wheel \
  protobuf-compiler libprotobuf-dev

export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

./build.sh \
  --config Release \
  --update --build \
  --build_shared_lib \
  --parallel $(nproc) \
  --skip_tests \
  --use_cuda \
  --use_tensorrt \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=87 \
    onnxruntime_BUILD_UNIT_TESTS=OFF
```

Install the output into a compact directory:

```bash
mkdir -p ~/onnxruntime-trt-1.14.1/lib ~/onnxruntime-trt-1.14.1/include

cp -a ~/onnxruntime-trt/build/Linux/Release/libonnxruntime.so* \
      ~/onnxruntime-trt-1.14.1/lib/
cp -a ~/onnxruntime-trt/build/Linux/Release/libonnxruntime_providers_*.so \
      ~/onnxruntime-trt-1.14.1/lib/
cp -a ~/onnxruntime-trt/include/onnxruntime/core/session/*.h \
      ~/onnxruntime-trt-1.14.1/include/

cd ~/onnxruntime-trt-1.14.1/lib
ln -sf libonnxruntime.so.* libonnxruntime.so
```

Rebuild only OmniXtreme against that ORT directory, then copy it over the default
CPU/auto binary in the normal `build/` tree:

```bash
cd ~/action_go
cmake -S . -B build_omni_trt \
  -Drealsense2_DIR=/usr/local/lib/cmake/realsense2 \
  -DACTION_GO_BUILD_OMNIXTREME=ON \
  -DACTION_GO_ONNXRUNTIME_DIR=$HOME/onnxruntime-trt-1.14.1
cmake --build build_omni_trt --target generic_composite_policy_player --parallel $(nproc)
cp -a build_omni_trt/generic_composite_policy_player build/generic_composite_policy_player

env -u LD_LIBRARY_PATH ldd ~/action_go/build/generic_composite_policy_player | grep onnxruntime
```

#### Ubuntu 22.04 / JetPack 6.x / TensorRT 10

ORT v1.14.1 is for the CUDA 11 / TensorRT 8.x generation.  On JetPack 6.x
(CUDA 12.x + TensorRT 10.x), TensorRT API errors such as `RNNOperation`,
`IFullyConnectedLayer`, `buildEngineWithConfig`, `getNbBindings`, or
`enqueueV2` mean the ORT/TRT versions are incompatible; switching only to
`--use_tensorrt_builtin_parser` is not enough.

For JetPack 6.2.1 (Ubuntu 22.04, CUDA 12.6, TensorRT 10.3), build a newer ORT.
Start with `v1.19.2` for the closest CUDA 12 / TensorRT 10 matrix row; if you
upgrade TensorRT to 10.4+, use ORT `v1.20.x` instead.  Check the official ORT
TensorRT compatibility matrix before changing versions:
https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

```bash
cd ~
export ORT_TAG=v1.19.2
export ORT_INSTALL=$HOME/onnxruntime-trt-${ORT_TAG#v}

git clone --recursive https://github.com/microsoft/onnxruntime.git onnxruntime-trt-jp6
cd ~/onnxruntime-trt-jp6
git checkout $ORT_TAG
git submodule update --init --recursive

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git build-essential libopenblas-dev \
  python3-pip python3-dev python3-setuptools python3-wheel \
  protobuf-compiler libprotobuf-dev

python3 -m pip install --user -U 'cmake>=3.28,<4'
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

./build.sh \
  --cmake_path $HOME/.local/bin/cmake \
  --config Release \
  --update --build \
  --build_shared_lib \
  --parallel $(nproc) \
  --skip_tests \
  --use_cuda \
  --use_tensorrt \
  --cuda_home /usr/local/cuda \
  --cudnn_home /usr/lib/aarch64-linux-gnu \
  --tensorrt_home /usr/lib/aarch64-linux-gnu \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=87 \
    onnxruntime_BUILD_UNIT_TESTS=OFF
```

If configure fails on Eigen hash verification with exactly this pair:

```text
expected: be8be39fdbc6e60e94fa7870b280707069b5b81a
actual:   32b145f525a8308d7ab1c09388b2e288312d8eba
```

patch the stale SHA1 in that ORT branch and rerun the same `build.sh` command:

```bash
sed -i 's/be8be39fdbc6e60e94fa7870b280707069b5b81a/32b145f525a8308d7ab1c09388b2e288312d8eba/' cmake/deps.txt
```

After configure succeeds, a manual resume is also fine; adjust the path to
your ORT checkout:

```bash
cmake --build build/Linux/Release -- -j4
# example from the robot log:
# cmake --build "$ONNXRUNTIME_TRT_ROOT/build/Linux/Release" -- -j4
```

On JetPack 6.2.1 this full build was validated with `-j4` and took about
71 minutes.  `clog.c` `write(...)` warnings, CMake policy/deprecation warnings,
and the probe message `nvcc fatal: Unknown option '-Wstrict-aliasing'` were
non-blocking as long as the final build exits with 0.

```bash
mkdir -p $ORT_INSTALL/lib $ORT_INSTALL/include
cp -a build/Linux/Release/libonnxruntime.so* $ORT_INSTALL/lib/
cp -a build/Linux/Release/libonnxruntime_providers_*.so $ORT_INSTALL/lib/
cp -a include/onnxruntime/core/session/*.h $ORT_INSTALL/include/
(cd $ORT_INSTALL/lib && ln -sf libonnxruntime.so.* libonnxruntime.so)
```

Then rebuild only OmniXtreme against the JP6 ORT directory and copy it over the
default CPU/auto binary in the normal `build/` tree:

```bash
cd ~/action_go
cmake -S . -B build_omni_trt \
  -Drealsense2_DIR=/usr/local/lib/cmake/realsense2 \
  -DACTION_GO_BUILD_OMNIXTREME=ON \
  -DACTION_GO_ONNXRUNTIME_DIR=$HOME/onnxruntime-trt-1.19.2
cmake --build build_omni_trt --target generic_composite_policy_player --parallel $(nproc)
cp -a build_omni_trt/generic_composite_policy_player build/generic_composite_policy_player

env -u LD_LIBRARY_PATH ldd build/generic_composite_policy_player | grep onnxruntime
```

Start with the default launcher mode (`ACTION_GO_ORT_MODE=clean`), which removes
ONNX Runtime entries from `LD_LIBRARY_PATH` and lets every binary use its own
RPATH.  Do **not** export `ACTION_GO_ORT=$HOME/onnxruntime-trt-1.19.2` for the
split build, because that would force BFM-Zero/BeyondMimic/etc. to load the TRT
ORT too.

Common issues:

| Symptom | Likely cause | Fix |
|---|---|---|
| `CMake 3.24 or higher is required` | Jetson `/usr/bin/cmake` is too old, e.g. 3.22.1 | `python3 -m pip install --user -U 'cmake>=3.24,<4'`; then `export PATH=$HOME/.local/bin:$PATH`, verify `which cmake && cmake --version`, remove the failed ORT build dir, and rerun `build.sh` (or pass `--cmake_path $HOME/.local/bin/cmake`) |
| Eigen SHA1 mismatch: expected `be8be39...`, actual `32b145...` | The ORT branch has a stale Eigen hash in `cmake/deps.txt` | Patch `cmake/deps.txt` as shown in the JP6 section, then rerun configure/build |
| TensorRT API compile errors: `RNNOperation`, `ResizeMode`, `SliceMode`, `IFullyConnectedLayer`, `buildEngineWithConfig`, `getNbBindings`, `enqueueV2`, `destroy` | ORT v1.14.1 is being built against TensorRT 10.x / CUDA 12.x on Ubuntu 22.04 or JetPack 6.x | Use JetPack 5.x for ORT v1.14.1, or use the Ubuntu 22.04 / JP6 recipe above with ORT v1.19.2+; `--use_tensorrt_builtin_parser` alone does not fix this |
| `build/generic_composite_policy_player` is not generated | CMake did not find ORT lib/include, or ONNX player targets were disabled | Configure with a valid `-DACTION_GO_ONNXRUNTIME_DIR=...`, then build `action_go_all` or target `generic_composite_policy_player` |
| `libonnxruntime.so.1: version 'VERS_1.19.2' not found` | Runtime `LD_LIBRARY_PATH` loaded a different ORT than the binary was linked against | Use the default launcher clean mode, or test with `env -u LD_LIBRARY_PATH ldd build/<player> | grep onnxruntime`; only use `ACTION_GO_ORT_MODE=prepend` when all players share one ORT |
| `TensorrtExecutionProvider` is missing | Linked to CPU-only ORT | Check `LD_LIBRARY_PATH`, `ldd`, and `libonnxruntime_providers_tensorrt.so` |
| `ModelProto does not have a graph` | ONNX file is empty or Git LFS pointer was not pulled | Run `git lfs pull`; check `ls -lh actions/omnixtreme/model/*.onnx` |
| `infer_ms` around 20 ms and `infer_late` grows | CPU EP is being used | Use `--ep tensorrt`; verify provider list |
| First run takes seconds or ready timeout | TensorRT engine build / lazy warmup | Use `--trt-cache-dir`, increase prestart timeout, and keep `--warmup-iters 1` |
| INT64 / subnormal FP16 warnings | TensorRT conversion warnings | Usually non-fatal; compare `--no-trt-fp16` only if outputs look numerically wrong |

Recommended real-robot runtime settings:

```bash
--ep tensorrt
--trt-fp16
--trt-cache-dir "$ACTION_GO_ROOT/.cache/omnixtreme_trt"
--onnx-threads 1
--hz 50
--publish-hz 500
--warmup-iters 1
--initial-noise-scale 1.0
--prepare-ramp-sec 0.1
--prepare-hold-sec 0.05
--startup-interp-sec 0.0
--target-interp-sec 0.0
--no-friction-feedforward
```

Notes:

- `initial_noise` should be randomly sampled for the flow policy.
- Do not use `--deterministic` unless debugging repeatability.
- Avoid global target interpolation for high-dynamic clips; even a few
  milliseconds of smoothing can introduce phase lag.
- `deploy_mujoco.py`'s friction term is a simulator actuator-loss model; do not
  blindly send it as real-robot torque feedforward.
- Mild motor sound can remain if the real motor model differs from the model
  used during training.

## Strict latency benchmark against the OmniXtreme paper

The OmniXtreme paper reports about `10 ms` onboard latency.  To compare
strictly, do not compare that number directly with the player's narrow
`infer_ms` unless the timing scope is identical.

Report at least these two scopes:

1. **Steady-state policy-loop inference latency**
   - Starts from a fresh robot-state snapshot.
   - Includes observation construction, FK ONNX, base-flow ONNX, residual ONNX,
     action scaling/clipping, and target construction.
   - Excludes process startup, TensorRT engine build, UserCtrl handoff, and DDS
     transport.
   - In current logs this is `infer_ms(last/avg/max)`.

2. **End-to-end LowState-to-LowCmd latency**
   - Starts when the `LowState` sample used by the policy is received.
   - Ends after the first `LowCmd` containing the newly inferred target is
     written by the 500 Hz publisher.
   - In current logs this is `e2e_ms(last/avg/max,n)`.

For a paper-level comparison, use the second number as the primary number, and
report the first number as an internal breakdown.

### Reproducible benchmark setup

On the robot:

```bash
sudo nvpmodel -m 0    # if supported by the robot image
sudo jetson_clocks    # if supported by the robot image
# Keep ONNX Runtime paths out of LD_LIBRARY_PATH for split builds; use binary RPATH.
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

Keep fixed:

- robot and onboard computer,
- JetPack / CUDA / TensorRT / ONNXRuntime versions,
- TensorRT precision (`--trt-fp16` or FP32),
- TensorRT cache state,
- motion segment,
- policy frequency (`50 Hz`),
- publish frequency (`500 Hz`),
- warmup policy (`--warmup-iters 1`).

Collect both:

- **cold startup**: remove the TensorRT cache and report session/engine build
  time separately;
- **warm steady-state**: reuse cache and warm up before measuring control-loop
  latency.

Example command:

```bash
cd ~/action_go
./build/generic_composite_policy_player \
  --spec actions/omnixtreme/config/generic_omnixtreme.yaml \
  --net eth0 \
  --cmd-topic rt/user_lowcmd \
  --trajectory actions/omnixtreme/motions/segments/11_side_flip_roll.npz \
  --ep tensorrt \
  --trt-fp16 \
  --trt-cache-dir ~/action_go/.cache/omnixtreme_trt \
  --onnx-threads 1 \
  --hz 50 \
  --publish-hz 500 \
  --warmup-iters 1 \
  --max-frames 364 \
  --stats-interval 1 \
  --dry-run
```

For a real-robot run, remove `--dry-run` and run through the action_go remote
supervisor so UserCtrl handoff is also exercised.

Example log fields:

```text
infer_ms(last/avg/max)=4.86/4.92/5.91
e2e_ms(last/avg/max,n)=...
infer_late=0
pub_late=0
pub_late_max_ms=0
```

Recommended table:

| Scope | EP | Precision | Cache | Warmup | Mean | P50 | P95 | P99 | Max | Deadline misses |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Policy-loop inference (`infer_ms`) | TensorRT | FP16 | warm | yes | | | | | | |
| LowState-to-LowCmd (`e2e_ms`) | TensorRT | FP16 | warm | yes | | | | | | |
| CPU baseline | CPU | FP32 | n/a | n/a | | | | | | |

Also report:

- `infer_late`: 50 Hz inference deadline misses;
- `pub_late` and `pub_late_max_ms`: 500 Hz publisher deadline misses;
- prestart ready time;
- first warmup inference time;
- cold TensorRT engine build time.

## action_go remote supervisor

`config/motion_actions.yaml` registers `omni_motion_default` and all split segments.  In dry-run mode:

```bash
tools/launch/start_dry_run.sh
```

Real robot execution should be staged: `--check-only`, sim viewer, action_go dry-run,
short `max_frames`, then full action.

For OmniXtreme split segments, per-action IMU safety is disabled
(`safety_enable: false`).  `Select+Start` still forces official passive/damping
and returns to shell.  All segments switch back to official `walkrun` when
finished, except segment 15, which switches back to official passive/damping.

Remote bindings:

- `F1+X`: `omni_seg04_wbst_stand` -> segment 4, `4_wbst_stand.npz`
- `F1+Y`: `omni_seg12_wbst_motive` -> segment 12, `12_wbst_motive.npz`
- `F1+B`: `omni_seg11_side_flip_roll` -> segment 11, `11_side_flip_roll.npz`

On the robot, make sure the ONNX files are real files under
`$ACTION_GO_ROOT/actions/omnixtreme/model/`. In a development tree they may be
symlinks to `$OMNIXTREME_ROOT/policy`, so deploy with symlink
dereferencing, for example:

```bash
rsync -avL actions/omnixtreme/model/ \
  unitree@ROBOT:~/action_go/actions/omnixtreme/model/
```

Quick robot-side check:

```bash
ls -lh "$ACTION_GO_ROOT/actions/omnixtreme/model/base_policy_trt.onnx" \
       "$ACTION_GO_ROOT/actions/omnixtreme/model/residual_policy.onnx" \
       "$ACTION_GO_ROOT/actions/omnixtreme/model/fk_trt.onnx"
```

## Split segments

The concatenated `motion.npz` has been split into standalone, renamed segment
motions in `actions/omnixtreme/motions/segments/`, with action entries in
`actions/omnixtreme/omnixtreme_segments.yaml`.  Canonical action names follow
the renamed file stems.  Compatibility aliases like `omni_seg04_1728_2193` are
also kept.

| Action | Motion file | max_frames |
| --- | --- | ---: |
| `omni_seg01_fuck_floor` | `motions/segments/1_fuck_floor.npz` | 850 |
| `omni_seg02_back_hand_flip_roll` | `motions/segments/2_back_hand_flip_roll.npz` | 485 |
| `omni_seg03_back_hand_flip` | `motions/segments/3_back_hand_flip.npz` | 393 |
| `omni_seg04_wbst_stand` | `motions/segments/4_wbst_stand.npz` | 465 |
| `omni_seg05_side_roll_kick` | `motions/segments/5_side_roll_kick.npz` | 414 |
| `omni_seg06_front_roll` | `motions/segments/6_front_roll.npz` | 503 |
| `omni_seg07_back_roll` | `motions/segments/7_back_roll.npz` | 482 |
| `omni_seg08_none` | `motions/segments/8_none.npz` | 123 |
| `omni_seg09_back_hand_flip` | `motions/segments/9_back_hand_flip.npz` | 587 |
| `omni_seg10_back_hand_flip` | `motions/segments/10_back_hand_flip.npz` | 473 |
| `omni_seg11_side_flip_roll` | `motions/segments/11_side_flip_roll.npz` | 364 |
| `omni_seg12_wbst_motive` | `motions/segments/12_wbst_motive.npz` | 573 |
| `omni_seg13_ceshoufan_double` | `motions/segments/13_ceshoufan_double.npz` | 410 |
| `omni_seg14_yewendun_kick` | `motions/segments/14_yewendun-kick.npz` | 916 |
| `omni_seg15_front_paxing` | `motions/segments/15_front_paxing.npz` | 429, returns passive |

`omni_motion_default` currently points to `omni_seg04_wbst_stand` for real-robot
remote testing.  To change the default later, update only this alias and keep
the remote binding unchanged.

## Crop motion

Use Python slice convention `[start, end)`.  To include frame 850, set
`--end-frame 851` or use `--end-frame-inclusive 850`:

```bash
tools/export/omnixtreme_crop_motion.py \
  --input actions/omnixtreme/motions/motion.npz \
  --output actions/omnixtreme/motions/segments/custom_segment.npz \
  --start-frame 0 \
  --end-frame 851 \
  --overwrite
```
