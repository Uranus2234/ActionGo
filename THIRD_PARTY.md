# Third-Party Components and Policy Assets

ActionGo's Apache-2.0 license covers ActionGo-owned code and documentation. It
does not automatically cover third-party SDKs, libraries, model weights,
motions, datasets, tags, product names, or upstream code.

## Runtime dependencies

| Component | Use | Distribution policy |
|---|---|---|
| Unitree SDK2 | DDS and G1 controller APIs | Installed separately; follow the upstream license |
| ONNX Runtime 1.22.0 | Policy inference | Download separately; retain Microsoft's license and notices |
| yaml-cpp, Eigen, zlib, OpenSSL, nlohmann/json | Runtime support | Installed from the system; follow each upstream license |
| MuJoCo, OpenCV, RealSense, AprilTag, ZeroMQ | Optional policy/simulation support | Installed separately; follow each upstream license |

Pinned reference versions are documented in `DEPENDENCIES.md`. The source-only
release does not redistribute precompiled ONNX Runtime libraries.

## Policy adapters and assets

ActionGo contains adapters for multiple upstream research policies. An adapter
does not grant permission to copy its corresponding weights or motion data.
`MODEL_ASSETS.json` is the machine-readable provenance and release gate. An
asset marked `blocked_pending_review` must not be included in a public release.

When adding an asset, record its exact source revision, asset-specific license,
SHA256, and redistribution decision. Source-code license information alone is
not sufficient evidence for model or dataset redistribution.
