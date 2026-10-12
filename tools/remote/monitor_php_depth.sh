#!/usr/bin/env bash
set -euo pipefail

unset LD_LIBRARY_PATH LD_PRELOAD

action_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
robot_host="${PHP_ROBOT_HOST:-unitree@192.168.3.162}"
robot_root="${PHP_ROBOT_ROOT:-/home/unitree/action_go}"
robot_net="${PHP_ROBOT_NET:-eth0}"
control_path="${PHP_SSH_CONTROL_PATH:-/tmp/php-robot-live-control}"
monitor_python="${PHP_MONITOR_PYTHON:-}"
if [[ -z "$monitor_python" ]]; then
    if [[ -x "$action_root/../php_parkour/.venv-php-sim/bin/python" ]]; then
        monitor_python="$action_root/../php_parkour/.venv-php-sim/bin/python"
    else
        monitor_python=python3
    fi
fi

if ! ssh -S "$control_path" -O check "$robot_host" 2>/dev/null; then
    printf '连接机器人；如提示密码，请输入机器人 SSH 密码。\n'
    ssh -M -N -f -S "$control_path" -o ControlPersist=600 -o ConnectTimeout=10 "$robot_host"
fi
scp -o ControlPath="$control_path" -o BatchMode=yes \
    "$action_root/tools/sim/php_depth_live.py" "$robot_host:/tmp/actiongo_php_depth_live.py"

printf '旁路监控 PHP 策略实际深度及角度；不启动策略、不占相机，关闭窗口不停止策略。\n'
exec "$monitor_python" -u "$action_root/tools/sim/php_depth_live.py" \
    --policy --host "$robot_host" --root "$robot_root" --net "$robot_net" \
    --remote-script /tmp/actiongo_php_depth_live.py --control-path "$control_path" \
    --output "$action_root/outputs/php_depth_live" "$@"
