param(
    [string]$Version = "1.1.0",
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $BuildOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    $BuildOutputDirectory = [System.IO.Path]::GetFullPath((Join-Path $Root $OutputDirectory))
}
$Exe = Join-Path $BuildOutputDirectory "PYsemantic.exe"
$Guide = Join-Path $Root "docs\PYsemantic操作指南.pdf"
$ReleaseGuide = Join-Path $BuildOutputDirectory "PYsemantic-User-Guide.pdf"
$PackageName = "PYsemantic-Windows-x64-v$Version"
$StageRoot = [System.IO.Path]::GetFullPath((Join-Path $Root "build\portable-package"))
$Stage = [System.IO.Path]::GetFullPath((Join-Path $StageRoot $PackageName))
$ZipPath = Join-Path $BuildOutputDirectory "$PackageName.zip"

foreach ($RequiredFile in @(
    $Exe,
    $Guide,
    (Join-Path $Root "docs\README-快速开始.txt"),
    (Join-Path $Root "METHOD_NOTES.md"),
    (Join-Path $Root "references.json"),
    (Join-Path $Root "THIRD_PARTY_NOTICES.txt"),
    (Join-Path $Root "examples\synthetic_texts.csv")
)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required portable package file is missing: $RequiredFile"
    }
}

New-Item -ItemType Directory -Path $BuildOutputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $StageRoot -Force | Out-Null
$StageRootPrefix = $StageRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $Stage.StartsWith($StageRootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to stage files outside build\portable-package."
}
if (Test-Path -LiteralPath $Stage) {
    Remove-Item -LiteralPath $Stage -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $Stage "examples") -Force | Out-Null

Copy-Item -LiteralPath $Exe -Destination (Join-Path $Stage "PYsemantic.exe")
Copy-Item -LiteralPath $Guide -Destination (Join-Path $Stage "PYsemantic操作指南.pdf")
Copy-Item -LiteralPath $Guide -Destination $ReleaseGuide -Force
Copy-Item -LiteralPath (Join-Path $Root "docs\README-快速开始.txt") -Destination (Join-Path $Stage "README-快速开始.txt")
Copy-Item -LiteralPath (Join-Path $Root "METHOD_NOTES.md") -Destination (Join-Path $Stage "方法说明.md")
Copy-Item -LiteralPath (Join-Path $Root "references.json") -Destination (Join-Path $Stage "来源文献.json")
Copy-Item -LiteralPath (Join-Path $Root "THIRD_PARTY_NOTICES.txt") -Destination (Join-Path $Stage "第三方许可.txt")
Copy-Item -LiteralPath (Join-Path $Root "examples\synthetic_texts.csv") -Destination (Join-Path $Stage "examples\synthetic_texts.csv")

$ManifestLines = Get-ChildItem -LiteralPath $Stage -File -Recurse | Sort-Object FullName | ForEach-Object {
    $Relative = [System.IO.Path]::GetRelativePath($Stage, $_.FullName).Replace("\", "/")
    $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $Relative"
}
[System.IO.File]::WriteAllLines((Join-Path $Stage "SHA256SUMS.txt"), $ManifestLines, [System.Text.UTF8Encoding]::new($false))

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath -CompressionLevel Optimal -Force
$StageFullPath = [System.IO.Path]::GetFullPath($Stage)
$WorkspacePrefix = [System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $StageFullPath.StartsWith($WorkspacePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove a staging directory outside the project workspace."
}
Remove-Item -LiteralPath $StageFullPath -Recurse -Force

$ZipHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ExeHash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash.ToLowerInvariant()
$GuideHash = (Get-FileHash -LiteralPath $Guide -Algorithm SHA256).Hash.ToLowerInvariant()
$SidecarLines = @("$ZipHash  $PackageName.zip", "$ExeHash  PYsemantic.exe", "$GuideHash  PYsemantic-User-Guide.pdf")
[System.IO.File]::WriteAllLines((Join-Path $BuildOutputDirectory "SHA256SUMS.txt"), $SidecarLines, [System.Text.UTF8Encoding]::new($false))
Write-Host "Portable package: $ZipPath"
Write-Host "SHA256: $ZipHash"
