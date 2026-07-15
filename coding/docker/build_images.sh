#!/usr/bin/env bash
# Build a per-task Docker image for each owned coding task. Each task dir is the
# build context; the shared language Dockerfile copies that task's hidden tests in.
set -euo pipefail
cd "$(dirname "$0")/.."

LANG_DOCKERFILE=(
  python:python.Dockerfile
  rust:rust.Dockerfile
  javascript:javascript.Dockerfile
)

for entry in "${LANG_DOCKERFILE[@]}"; do
  language="${entry%%:*}"
  dockerfile="${entry##*:}"
  for task_dir in coding/tasks/*-"$language"; do
    [ -d "$task_dir" ] || continue
    task_id="$(basename "$task_dir")"
    docker build -f "coding/docker/$dockerfile" -t "bench/$task_id" "$task_dir"
  done
done
