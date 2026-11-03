# Reproducible Dependencies

The reference x86_64 build is Ubuntu 22.04, GCC 11, and CMake 3.22. The robot
build uses Jetson/JetPack versions compatible with the selected TensorRT and
ONNX Runtime builds.

| Dependency | Reference version | Purpose |
|---|---|---|
| C++ compiler | GCC 11.4, C++17 | Runtime and tools |
| CMake | 3.22.1; 3.24+ for custom ORT/TensorRT builds | Build system |
| Unitree SDK2 | commit `21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b` | G1 DDS and controller APIs |
| yaml-cpp | 0.7.0 | Policy/action configuration |
| Eigen | 3.4.0 | Runtime math |
| zlib | 1.2.11 | NPZ and CRC support |
| OpenSSL | 3.0.x | HMAC authentication for optional UDP group control |
| nlohmann/json | 3.10+ | Signed UDP and SONIC protocol parsing |
| ONNX Runtime CPU | 1.22.0, bundled for x86_64/aarch64; official archives verified with pinned SHA256 | Generic players |
| ONNX Runtime + TensorRT | 1.19.2 reference build | OmniXtreme accelerated deployment |
| MuJoCo | 3.8.1 reference, optional | Humanoid-GPT FK and simulation |
| librealsense2 | deployment-specific, optional | Instinct/SteadyTray sensors |
| AprilTag C library | deployment-specific, optional | SteadyTray pose input |
| libzmq / Redis CLI | deployment-specific, optional | TWIST realtime input |

The complete official CPU SDKs are committed as regular Git files in
`thirdparty/onnxruntime-linux-x64-1.22.0/` and
`thirdparty/onnxruntime-linux-aarch64-1.22.0/`, including headers, shared libraries,
relative library symlinks, licenses, and third-party notices. Git LFS is not
required for these SDKs. CMake selects the host architecture automatically;
custom ONNX Runtime directory overrides remain supported.

To restore an SDK from its official archive:

```bash
tools/deps/fetch_onnxruntime.sh             # current architecture
tools/deps/fetch_onnxruntime.sh aarch64     # robot target archive
```

The downloader pins both the release URL and SHA256. The CI workflow pins
Unitree SDK2 and verifies the same CPU ONNX Runtime archive. Record
the full JetPack, CUDA, TensorRT, ONNX Runtime, camera firmware, robot firmware,
and kernel versions in every real-robot experiment manifest.

Do not globally prepend an unrelated ONNX Runtime through `LD_LIBRARY_PATH`.
The default launchers use each binary's RPATH; see `actions/tracking/omnixtreme/README.md`
for the separate TensorRT build.
