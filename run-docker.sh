#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo 'Usage: run-docker.sh INPUT OUTPUT [--meter 4/4 --bpm 120 --resume ...]' >&2
  exit 2
fi
INPUT_PATH="$(realpath -- "$1")"
mkdir -p -- "$2"
OUTPUT_PATH="$(realpath -- "$2")"
shift 2
CONTAINER_INPUT=/input
if [[ -f "$INPUT_PATH" ]]; then CONTAINER_INPUT="/input/$(basename -- "$INPUT_PATH")"; fi
GPU_ARGS=()
if [[ "${PIANO_DOCKER_GPU:-0}" == 1 ]]; then GPU_ARGS=(--gpus all); fi
# Only the chosen media and output tree are mounted; original media is read-only.
exec docker run --rm --user "$(id -u):$(id -g)" "${GPU_ARGS[@]}" \
  -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba \
  -v "$INPUT_PATH:$CONTAINER_INPUT:ro" -v "$OUTPUT_PATH:/output" \
  piano-transcription-skill:local "$CONTAINER_INPUT" --output /output --work-dir /output/.piano-work "$@"
