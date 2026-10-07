#!/usr/bin/env bash
set -euo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SKILL_DIR/.deployment.json" ]]; then
  "$SKILL_DIR/deploy.sh"
fi
exec "${PYTHON_BOOTSTRAP:-python3}" "$SKILL_DIR/scripts/launch.py" "$@"
