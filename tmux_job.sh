#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ACTION="${1:-help}"

usage() {
    cat <<'EOF'
用法:
  bash tmux_job.sh start <会话名> <日志路径> <命令> [参数...]
  bash tmux_job.sh attach <会话名>
  bash tmux_job.sh stop <会话名>
  bash tmux_job.sh status [会话名]

示例:
  bash tmux_job.sh start o2_r50 logs/o2_r50.log \
    bash run.sh train-r50 --fold 0 --device 0,1 --batch 4 --name o2_rtdetr_r50vd_fold0
EOF
}

case "$ACTION" in
    start)
        [[ $# -ge 5 ]] || { usage >&2; exit 2; }
        SESSION="$2"
        LOG_PATH="$3"
        shift 3
        command -v tmux >/dev/null || { echo "错误: 未安装 tmux" >&2; exit 1; }
        if tmux has-session -t "=${SESSION}" 2>/dev/null; then
            echo "错误: tmux 会话已存在: $SESSION" >&2
            exit 1
        fi
        [[ "$LOG_PATH" = /* ]] || LOG_PATH="$PROJECT/$LOG_PATH"
        mkdir -p "$(dirname "$LOG_PATH")"
        printf -v PROJECT_Q "%q" "$PROJECT"
        printf -v LOG_Q "%q" "$LOG_PATH"
        printf -v COMMAND_Q "%q " "$@"
        tmux new-session -d -s "$SESSION" \
            "cd $PROJECT_Q && $COMMAND_Q 2>&1 | tee -a $LOG_Q"
        echo "已启动 tmux 会话: $SESSION"
        echo "日志: $LOG_PATH"
        echo "查看: tmux attach -t $SESSION"
        ;;
    attach)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        exec tmux attach -t "=$2"
        ;;
    stop)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        tmux kill-session -t "=$2"
        echo "已停止 tmux 会话: $2"
        ;;
    status)
        if [[ $# -eq 2 ]]; then
            SESSIONS="$(tmux list-sessions -F '#{session_name} #{session_created_string} #{session_windows} windows' 2>/dev/null || true)"
            MATCH="$(awk -v target="$2" '$1 == target' <<<"$SESSIONS")"
            if [[ -n "$MATCH" ]]; then
                echo "$MATCH"
            else
                echo "没有找到 tmux 会话: $2"
            fi
        else
            tmux list-sessions 2>/dev/null || echo "没有运行中的 tmux 会话"
        fi
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "错误: 不支持的操作: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
