#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  exec "$@"
fi

TARGET_UID="${STAD_CONTAINER_UID:-}"
TARGET_GID="${STAD_CONTAINER_GID:-}"
TARGET_USER="${STAD_CONTAINER_USER:-stad}"

if [ -z "$TARGET_UID" ] || [ -z "$TARGET_GID" ]; then
  exec "$@"
fi

group_name_from_gid() {
  getent group "$1" | cut -d: -f1
}

user_name_from_uid() {
  getent passwd "$1" | cut -d: -f1
}

home_from_uid() {
  getent passwd "$1" | cut -d: -f6
}

TARGET_GROUP="$(group_name_from_gid "$TARGET_GID" || true)"
if [ -z "$TARGET_GROUP" ]; then
  TARGET_GROUP="$TARGET_USER"
  if getent group "$TARGET_GROUP" >/dev/null 2>&1; then
    TARGET_GROUP="stad-${TARGET_GID}"
  fi
  groupadd --gid "$TARGET_GID" "$TARGET_GROUP" >/dev/null 2>&1
fi

EXISTING_USER="$(user_name_from_uid "$TARGET_UID" || true)"
if [ -z "$EXISTING_USER" ]; then
  NEW_USER="$TARGET_USER"
  if getent passwd "$NEW_USER" >/dev/null 2>&1; then
    NEW_USER="stad-${TARGET_UID}"
  fi
  useradd \
    --uid "$TARGET_UID" \
    --gid "$TARGET_GID" \
    --create-home \
    --shell /bin/bash \
    "$NEW_USER" >/dev/null 2>&1
  EXISTING_USER="$(user_name_from_uid "$TARGET_UID" || true)"
fi

if [ -z "$EXISTING_USER" ]; then
  echo "Failed to create a named container user for UID ${TARGET_UID}, continuing as root." >&2
  exec "$@"
fi

TARGET_HOME="$(home_from_uid "$TARGET_UID" || true)"
if [ -n "$TARGET_HOME" ]; then
  mkdir -p "$TARGET_HOME"
  chown "$TARGET_UID:$TARGET_GID" "$TARGET_HOME" || true
  export HOME="$TARGET_HOME"
fi

exec gosu "$TARGET_UID:$TARGET_GID" "$@"
