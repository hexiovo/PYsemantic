$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$ModelManifest = Join-Path $Root "assets\bge-small-zh-v1.5\model-manifest.json"
$ModelDirectory = Join-Path $Root "assets\bge-small-zh-v1.5"
$Icon = Join-Path $Root "assets\app.ico"
$Exe = Join-Path $Root "dist\ChineseSemanticToolkit.exe"
$DeliveryDirectory = "C:\Users\DELL\Desktop\APP"
$DeliveryExe = Join-Path $DeliveryDirectory "ChineseSemanticToolkit.exe"

if (-not (Test-Path -LiteralPath $Python)) { throw "Project .venv is missing. Create it with Python 3.12 and install requirements-lock.txt." }
if (-not (Test-Path -LiteralPath $ModelManifest)) { throw "Pinned local BGE model is missing. Run scripts\download_model.py before packaging." }
if (-not (Test-Path -LiteralPath (Join-Path $Root "METHOD_NOTES.md"))) { throw "METHOD_NOTES.md is missing." }
if (-not (Test-Path -LiteralPath (Join-Path $Root "references.json"))) { throw "references.json is missing." }
if (-not (Test-Path -LiteralPath (Join-Path $Root "THIRD_PARTY_NOTICES.txt"))) { throw "THIRD_PARTY_NOTICES.txt is missing." }

& $Python scripts\build_icon.py
if ($LASTEXITCODE -ne 0) { throw "Icon generation failed." }

$PyInstallerArgs = @(
    "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
    "--name", "ChineseSemanticToolkit",
    "--icon", $Icon,
    "--distpath", (Join-Path $Root "dist"),
    "--workpath", (Join-Path $Root "build\pyinstaller"),
    "--specpath", (Join-Path $Root "build"),
    "--add-data", "$ModelDirectory;assets\bge-small-zh-v1.5",
    "--add-data", "$Icon;assets",
    "--add-data", "$(Join-Path $Root 'METHOD_NOTES.md');.",
    "--add-data", "$(Join-Path $Root 'references.json');.",
    "--add-data", "$(Join-Path $Root 'THIRD_PARTY_NOTICES.txt');.",
    "--collect-all", "customtkinter",
    "--collect-all", "jieba",
    "--collect-all", "tokenizers",
    "--exclude-module", "torch",
    "--exclude-module", "matplotlib",
    "--exclude-module", "pytest",
    "--exclude-module", "IPython",
    "--exclude-module", "jupyter",
    "--exclude-module", "sklearn.tests",
    "--exclude-module", "scipy.tests",
    "--exclude-module", "numpy.tests",
    "--exclude-module", "onnxruntime.quantization",
    "--exclude-module", "transformers",
    (Join-Path $Root "main.py")
)
& $Python @PyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE." }
if (-not (Test-Path -LiteralPath $Exe)) { throw "Expected executable is missing: $Exe" }

New-Item -ItemType Directory -Path $DeliveryDirectory -Force | Out-Null
Copy-Item -LiteralPath $Exe -Destination $DeliveryExe -Force
$Info = Get-Item -LiteralPath $DeliveryExe
Write-Host "Packaged executable: $DeliveryExe"
Write-Host "Bytes: $($Info.Length)"

