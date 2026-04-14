#!/usr/bin/env sh

if [ -n "${ZSH_VERSION:-}" ]; then
  eval '_stad_script_path=${(%):-%N}'
elif [ -n "${BASH_SOURCE:-}" ]; then
  _stad_script_path="${BASH_SOURCE[0]}"
else
  _stad_script_path="$0"
fi

export STAD_PROJECT_ROOT="$(cd "$(dirname "$_stad_script_path")" && pwd)"
export STAD_DOCKER_IMAGE="${STAD_DOCKER_IMAGE:-stad:latest}"
export STAD_DOCKER_WORKDIR="${STAD_DOCKER_WORKDIR:-/workspace}"
export STAD_DOCKER_SHM_SIZE="${STAD_DOCKER_SHM_SIZE:-8g}"
export STAD_DOCKER_GPU="${STAD_DOCKER_GPU:-0}"
export STAD_WANDB_MODE="${STAD_WANDB_MODE:-offline}"
export STAD_AUTO_SHELL="${STAD_AUTO_SHELL:-1}"
export STAD_DOCKER_USER_NAME="${STAD_DOCKER_USER_NAME:-$(id -un)}"

_stad_require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not on PATH" >&2
    return 1
  fi
}

_stad_require_docker_daemon() {
  _stad_require_docker || return 1
  if ! docker info >/dev/null 2>&1; then
    echo "docker is installed, but the Docker daemon is not reachable. Start Docker Desktop or your Docker service first." >&2
    return 1
  fi
}

_stad_docker_gpu_args() {
  if [ "$STAD_DOCKER_GPU" = "1" ]; then
    printf '%s\n' "--gpus all"
  fi
}

stad_doctor() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker: missing"
    return 1
  fi

  echo "docker: $(docker --version)"
  if docker info >/dev/null 2>&1; then
    echo "daemon: reachable"
  else
    echo "daemon: unreachable"
    return 1
  fi
}

stad_build() {
  _stad_require_docker_daemon || return 1
  docker build \
    -t "$STAD_DOCKER_IMAGE" \
    -f "$STAD_PROJECT_ROOT/Dockerfile" \
    "$STAD_PROJECT_ROOT"
}

stad_run() {
  _stad_require_docker_daemon || return 1

  _stad_gpu_args="$(_stad_docker_gpu_args)"
  _stad_tty_args=""

  if [ -t 0 ] && [ -t 1 ]; then
    _stad_tty_args="-it"
  fi

  if [ "$#" -eq 0 ]; then
    set -- bash
  fi

  if [ -n "$_stad_gpu_args" ]; then
    docker run --rm $_stad_tty_args \
      $_stad_gpu_args \
      --shm-size "$STAD_DOCKER_SHM_SIZE" \
      -e STAD_CONTAINER_UID="$(id -u)" \
      -e STAD_CONTAINER_GID="$(id -g)" \
      -e STAD_CONTAINER_USER="$STAD_DOCKER_USER_NAME" \
      -e PYTHONPATH="$STAD_DOCKER_WORKDIR" \
      -e MPLBACKEND=Agg \
      -e WANDB_MODE="$STAD_WANDB_MODE" \
      -v "$STAD_PROJECT_ROOT:$STAD_DOCKER_WORKDIR" \
      -w "$STAD_DOCKER_WORKDIR" \
      "$STAD_DOCKER_IMAGE" \
      "$@"
  else
    docker run --rm $_stad_tty_args \
      --shm-size "$STAD_DOCKER_SHM_SIZE" \
      -e STAD_CONTAINER_UID="$(id -u)" \
      -e STAD_CONTAINER_GID="$(id -g)" \
      -e STAD_CONTAINER_USER="$STAD_DOCKER_USER_NAME" \
      -e PYTHONPATH="$STAD_DOCKER_WORKDIR" \
      -e MPLBACKEND=Agg \
      -e WANDB_MODE="$STAD_WANDB_MODE" \
      -v "$STAD_PROJECT_ROOT:$STAD_DOCKER_WORKDIR" \
      -w "$STAD_DOCKER_WORKDIR" \
      "$STAD_DOCKER_IMAGE" \
      "$@"
  fi
}

stad_shell() {
  stad_run bash
}

stad_preprocess() {
  stad_run python3 trainer.py preprocess "$@"
}

stad_train_mae() {
  stad_run python3 trainer.py mae "$@"
}

stad_train_stad() {
  stad_run python3 trainer.py stad "$@"
}

stad_eval() {
  stad_run python3 trainer.py eval "$@"
}

stad_test() {
  stad_run python3 -m pytest -q "$@"
}

_stad_is_interactive_shell() {
  case "$-" in
    *i*) return 0 ;;
    *) return 1 ;;
  esac
}

_stad_source_mode="${1:-}"
_stad_should_autolaunch=0

case "$_stad_source_mode" in
  --no-shell|no-shell|load)
    _stad_should_autolaunch=0
    ;;
  --shell|shell)
    _stad_should_autolaunch=1
    ;;
  "")
    if [ "$STAD_AUTO_SHELL" = "1" ] && _stad_is_interactive_shell; then
      _stad_should_autolaunch=1
    fi
    ;;
  *)
    echo "Unknown env.sh mode: $_stad_source_mode" >&2
    echo "Valid modes: --shell, --no-shell" >&2
    return 1
    ;;
esac

echo "STAD Docker environment loaded."
echo "Project root: $STAD_PROJECT_ROOT"
echo "Image tag: $STAD_DOCKER_IMAGE"
echo "Available commands: stad_doctor, stad_build, stad_shell, stad_run, stad_preprocess, stad_train_mae, stad_train_stad, stad_eval, stad_test"

if [ "$_stad_should_autolaunch" = "1" ]; then
  echo "Launching interactive Docker shell. Use 'source env.sh --no-shell' or 'export STAD_AUTO_SHELL=0' to only load helpers."
  stad_shell
fi
