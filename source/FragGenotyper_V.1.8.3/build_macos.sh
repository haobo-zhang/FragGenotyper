#!/bin/bash
# Run on the target macOS architecture; this script does not cross-compile.
set -euo pipefail

fg_directory="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
fg_python="${FRAGGENOTYPER_PYTHON:-python3}"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script requires macOS. On Windows run build_windows.ps1." >&2
    exit 1
fi

cd "$fg_directory"
"$fg_python" -c 'import importlib.util, platform, sys
required = ["PyInstaller", "PyQt6", "numpy", "pandas", "scipy", "matplotlib", "sklearn", "joblib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Missing build dependencies: " + ", ".join(missing) + "\nInstall requirements.txt and pyinstaller in the selected Python environment.")
print("Building with", sys.executable, "for", platform.machine())'

"$fg_python" -m PyInstaller \
    --noconfirm \
    --clean \
    --distpath "$fg_directory/dist" \
    --workpath "$fg_directory/build/macos" \
    "$fg_directory/FragGenotyper_V.1.8.3.spec"

echo "Built: $fg_directory/dist/FragGenotyper_V.1.8.3.app"
echo "The app is ad-hoc signed by the local build process; distribution signing/notarization is a separate step."
