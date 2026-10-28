#!/bin/bash
# Source launcher. Set FRAGGENOTYPER_PYTHON to use a specific Python environment.
set -euo pipefail

fg_directory="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
fg_python="${FRAGGENOTYPER_PYTHON:-python3}"
cd "$fg_directory"

exec "$fg_python" "$fg_directory/FragGenotyper_V.1.8.3.py" "$@"
