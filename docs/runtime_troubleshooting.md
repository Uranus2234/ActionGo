# Runtime Troubleshooting

## ONNX Runtime

The x86_64 and aarch64 CPU SDKs are included in `thirdparty/` as regular Git
files. A normal checkout provides their headers and shared libraries. If an SDK
is missing or damaged, restore the pinned CPU release for the current architecture:

```bash
tools/deps/fetch_onnxruntime.sh
```

For a custom location:

```bash
cmake -S . -B build \
  -DACTION_GO_ONNXRUNTIME_DIR=/path/to/onnxruntime
```

The directory must contain `include/onnxruntime_cxx_api.h` and `lib/libonnxruntime.so*`. Launcher scripts default to `ACTION_GO_ORT_MODE=clean`, removing unrelated ONNX Runtime entries from `LD_LIBRARY_PATH` so each binary uses its build RPATH.

Verify linkage with:

```bash
env -u LD_LIBRARY_PATH ldd build/generic_policy_player | rg onnxruntime
env -u LD_LIBRARY_PATH ldd build/generic_motion_tracking_player | rg onnxruntime
```

If all players intentionally use one custom build, set `ACTION_GO_ORT_MODE=prepend` and `ACTION_GO_ORT` to that installation. Do not mix ABI-incompatible CPU and TensorRT provider libraries in one process.

## TensorRT

OmniXtreme and SONIC can use TensorRT, but the compatible ONNX Runtime build depends on JetPack, CUDA, cuDNN, and TensorRT versions. Build those targets against a separate provider-enabled ORT by setting `ACTION_GO_MOTION_ONNXRUNTIME_DIR` or the adapter-specific build directory described in its README.

Before robot use, record the complete JetPack/CUDA/TensorRT/ORT stack, run the adapter `--check-only`, and confirm `ldd` resolves the intended libraries. A model loading under CPU ORT does not prove TensorRT engine compatibility or control-loop timing.
