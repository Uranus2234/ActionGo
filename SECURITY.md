# Security Policy

ActionGo can publish low-level commands to a real humanoid robot. Treat a
security bug as a possible physical-safety bug.

## Supported versions

Security fixes are provided for the latest tagged release and the current
default branch. Development branches are not supported deployment artifacts.

## Reporting

Do not open a public issue for vulnerabilities that could cause unauthorized
robot control, bypass command filtering, expose credentials, or make unsafe
handoff possible. Use GitHub's private vulnerability reporting for this
repository. If it is unavailable, contact the maintainer through the private
address listed on the maintainer's GitHub profile and request an encrypted
reporting channel.

Include affected revision, deployment topology, reproduction steps, expected
and observed behavior, and whether a real robot was involved. Do not test on a
robot or network you do not own or have permission to use.

## Trust boundaries

- Action YAML is trusted executable configuration. Its `command` field is run
  by a shell; do not load configuration from untrusted contributors or hosts.
- UDP group control is disabled by default. Legacy text commands are accepted
  only on loopback. Non-loopback and multicast listeners require signed JSON
  commands and a local key file that is never committed.
- Policy files, motion data, sensor streams, and ONNX models are untrusted
  inputs until their provenance and semantics have been verified.
- ActionGo is not a safety-certified controller. Network authentication and
  command filters do not replace physical barriers, an operator, or the
  robot's official passive/damping controls.
