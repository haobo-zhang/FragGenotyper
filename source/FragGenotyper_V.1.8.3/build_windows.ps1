# Build a native Windows one-file EXE. This script does not cross-compile.
# Windows execution and the generated EXE must be verified on Windows.
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "This script requires Windows. On macOS run build_macos.sh."
}

$fgDirectory = Split-Path -LiteralPath $PSCommandPath -Parent
Set-Location -LiteralPath $fgDirectory

$fgDependencyProbe = @'
import importlib.util
import platform
import sys
required = ["PyInstaller", "PyQt6", "numpy", "pandas", "scipy", "matplotlib", "sklearn", "joblib"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print("Python:", sys.executable, "Architecture:", platform.machine())
if missing:
    sys.exit("Missing dependencies: " + ", ".join(missing) + ". Install requirements.txt and pyinstaller.")
'@
# Standard input avoids native command-line quote differences between PowerShell versions.
$fgDependencyProbe | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "Dependency check failed for the selected Python environment."
}

$fgArguments = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", "FragGenotyper_V.1.8.3",
    "--paths", $fgDirectory,
    "--distpath", (Join-Path $fgDirectory "dist"),
    "--workpath", (Join-Path $fgDirectory "build/windows"),
    "--specpath", (Join-Path $fgDirectory "build/windows-spec"),
    "--collect-all", "scipy",
    "--collect-all", "sklearn",
    "--collect-submodules", "numpy",
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--runtime-hook", (Join-Path $fgDirectory "runtime_cache.py"),
    "--exclude-module", "PySide6",
    "--exclude-module", "PySide2",
    "--exclude-module", "PyQt5",
    "--exclude-module", "tkinter",
    "--exclude-module", "torch",
    "--exclude-module", "tensorflow",
    "--exclude-module", "jax",
    "--exclude-module", "jaxlib",
    "--exclude-module", "IPython",
    "--exclude-module", "notebook",
    "--add-data", ((Join-Path $fgDirectory "aq8s1-jliqi.icns") + ";.")
)

# The supplied .icns icon is for macOS. Use an optional native .ico on Windows.
$fgWindowsIcon = Join-Path $fgDirectory "aq8s1-jliqi.ico"
if (Test-Path -LiteralPath $fgWindowsIcon) {
    $fgArguments += @("--icon", $fgWindowsIcon)
}
$fgArguments += (Join-Path $fgDirectory "FragGenotyper_V.1.8.3.py")

$fgPreviousBackend = $env:MPLBACKEND
try {
    $env:MPLBACKEND = "QtAgg"
    & $Python @fgArguments
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
} finally {
    if ($null -eq $fgPreviousBackend) {
        Remove-Item Env:MPLBACKEND -ErrorAction SilentlyContinue
    } else {
        $env:MPLBACKEND = $fgPreviousBackend
    }
}

Write-Host ("Built: " + (Join-Path $fgDirectory "dist/FragGenotyper_V.1.8.3.exe"))
Write-Host "Verify this EXE on Windows with Qt, a genotype job, and figure export before distribution."
