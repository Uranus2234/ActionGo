# Runtime logging

The remote supervisor creates a private session directory automatically under
`ACTION_GO_LOG_DIR` (default `logs/`). `logs/latest` links to the latest session.
An explicit `--event-log` still selects the supervisor JSONL path.

## Console

The default interactive display refreshes at 5 Hz without scrolling:

```text
-----------------------------------------------
  STATE    USER_CTRL
  POLICY   kick_right    E2E  4.2 ms
-----------------------------------------------
```

`POLICY` is highlighted. Official `WALKRUN` and `DAMPING` show no policy or E2E
after the action exits. Switching and blocked handoffs are shown separately.
Initial state is `UNKNOWN`. Control state comes from successful controller RPCs,
not a guessed startup mode. It is the last acknowledged mode, not a continuous
firmware query: external mode changes are not detected. Lost LowState feedback
adds `[STALE]`. Dry-run displays `DRY_RUN`.

E2E measures local receipt of the exact state used by a command through
observation, inference, filtering and the first local DDS `Write` of that
command. Async mailboxes carry the state/command association. Repeated
publication does not create new samples. The latest completed measurement is
sampled at up to 5 Hz; no average or maximum is displayed. Samples older than
one second show `STALE`; unavailable measurements show `--`. This excludes robot
receipt/execution. Interpolation timing ends at the first publication beginning
to blend toward the new target, not completion of the blend.

All generated messages and labels are English. Noninteractive output uses plain
status lines. `ACTION_GO_CONSOLE=quiet` hides the panel but keeps notices;
`debug` mirrors captured output. Start/completion, recovery and errors remain
visible in normal mode. Existing config snapshots/raw output preserve their
original content without translation.

The tmux launcher preserves the terminal connection for the fixed display.
Its separate `action_go_*.log` files contain launcher/restart notices;
`tools/launch/start.sh logs` follows the current session's readable events.

## Session files

```text
<run_id>/
  session.json
  actions.yaml
  remote.yaml
  events.jsonl
  events.log
  session_events.jsonl
  session_events.log
  runtime.output.log
  summary.json
  actions/
    1_kick_right.jsonl
    1_kick_right.log
    1_kick_right.output.log
```

Metadata includes process/network, dry-run, config hashes and code revision
(`ACTION_GO_GIT_SHA`, set by the tmux launcher). Top-level configs are copied;
Generic spec players record their spec/model paths and SHA256 hashes.
Dedicated sensor/composite runtimes retain their existing model path events.
JSONL retains existing context/schema keys and adds `level` and `pid`. Each
stream has a readable UTC text log. Events cover action requests, handshakes,
controller results, process exits, safety failures and runtime metrics.

`command_timing` includes `state_seq`, `command_seq`, `state_wait_ms`,
`inference_ms`, `publish_wait_ms`, `e2e_ms`. Inference covers observation/target
generation plus model execution. Publish wait includes filtering and DDS
submission. Depth runtime DDS events use a separate `.dds.jsonl` stream to avoid
concurrent rotation of a shared file. Atomic `*.latest.json` files feed the panel.

`runtime.output.log` captures supervisor/SDK output; each child has its own raw
output in normal/quiet modes. In debug mode child output goes to the session
capture stream and console. `summary.json` is written on ordinary teardown,
including exception unwinding; absence indicates an interrupted session such as
SIGKILL. `*.health.json` records writer failure and dropped records. Check health
files even when shutdown was clean.

## Writing and retention

Structured logging uses background writers and persistent streams. Its bounded
queue reserves 128 additional error slots beyond the normal 1024. Producers do
not wait for capacity; overload is counted. This is buffered diagnostic logging,
not lossless recording. Explicit `flush()` supports checkpoints/tests; shutdown
drains queued records. Background flush occurs every 200 ms and on errors, without
per-event `fsync` durability.

Structured streams rotate at 32 MiB, keeping three previous segments;
`ACTION_GO_LOG_MAX_BYTES` adjusts the threshold. Supervisor raw output has the
same default limit. Child output files remain complete during an action, so
noisy children can exceed that per-stream limit.

The tmux launcher prunes inactive ActionGo sessions before each launch, keeping
up to seven days and targeting a 1 GiB total. Active sessions are skipped.
Retention runs between launches, not as a hard live quota; direct binary users
can run pruning explicitly. External `--event-log` paths are outside pruning.

```bash
python3 tools/logs/manage_logs.py --list
python3 tools/logs/manage_logs.py --errors
python3 tools/logs/manage_logs.py --run latest --action 1_kick_right
python3 tools/logs/manage_logs.py --export /path/on/data-disk/actiongo-debug.tar.gz
python3 tools/logs/manage_logs.py --prune --keep-days 7 --max-bytes 1073741824
tools/launch/start.sh logs
```

Queries merge events from all processes by timestamp. Exports contain captured
configuration and raw output; inspect them before sharing.
