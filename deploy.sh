#!/usr/bin/env bash
set -euo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BOOTSTRAP:-python3}" "$SKILL_DIR/scripts/deploy.py" "$@"
