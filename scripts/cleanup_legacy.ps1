[CmdletBinding()]
param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd([IO.Path]::DirectorySeparatorChar)
$scriptRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
$expectedScriptRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'scripts')).TrimEnd([IO.Path]::DirectorySeparatorChar)
$rootPrefix = $repoRoot + [IO.Path]::DirectorySeparatorChar

if ([string]::Compare($scriptRoot, $expectedScriptRoot, [StringComparison]::OrdinalIgnoreCase) -ne 0) {
    throw 'The cleanup script must run from the repository scripts directory.'
}
foreach ($marker in @('main.py', 'build.ps1', '.gitignore')) {
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot $marker) -PathType Leaf)) {
        throw "Repository marker missing: $marker"
    }
}

function Assert-ContainedPath {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = [IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a target outside the repository: $fullPath"
    }
    return $fullPath
}

function Assert-NoReparsePoints {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = Assert-ContainedPath $Path
    $rootItem = Get-Item -LiteralPath $fullPath -Force
    if ($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing a reparse-point cleanup target: $fullPath"
    }
    if (-not $rootItem.PSIsContainer) { return }

    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    $pending.Enqueue($fullPath)
    while ($pending.Count -gt 0) {
        $current = $pending.Dequeue()
        foreach ($entry in [IO.Directory]::GetFileSystemEntries($current)) {
            $item = Get-Item -LiteralPath $entry -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing to recurse through a junction or symlink: $($item.FullName)"
            }
            if ($item.PSIsContainer) { $pending.Enqueue($item.FullName) }
        }
    }
}

function Get-DirectoryStats {
    param([Parameter(Mandatory)][string]$Path)
    Assert-NoReparsePoints $Path
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    $pending.Enqueue([IO.Path]::GetFullPath($Path))
    [long]$bytes = 0
    [long]$files = 0
    [long]$directories = 0
    while ($pending.Count -gt 0) {
        $current = $pending.Dequeue()
        foreach ($entry in [IO.Directory]::GetFileSystemEntries($current)) {
            $item = Get-Item -LiteralPath $entry -Force
            if ($item.PSIsContainer) {
                $directories++
                $pending.Enqueue($item.FullName)
            } else {
                $files++
                $bytes += $item.Length
            }
        }
    }
    return [pscustomobject]@{ Bytes = $bytes; Files = $files; Directories = $directories }
}

function Find-CacheDirectories {
    param([Parameter(Mandatory)][string]$BasePath)
    $base = Assert-ContainedPath $BasePath
    if (-not (Test-Path -LiteralPath $base -PathType Container)) { return @() }
    Assert-NoReparsePoints $base
    $cacheNames = @('__pycache__', '.pytest_cache', '.mypy_cache')
    $found = New-Object 'System.Collections.Generic.List[string]'
    $pending = New-Object 'System.Collections.Generic.Queue[string]'
    $pending.Enqueue($base)
    while ($pending.Count -gt 0) {
        $current = $pending.Dequeue()
        foreach ($entry in [IO.Directory]::GetDirectories($current)) {
            $item = Get-Item -LiteralPath $entry -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                continue
            }
            if ($cacheNames -contains $item.Name) {
                $found.Add($item.FullName)
            } else {
                $pending.Enqueue($item.FullName)
            }
        }
    }
    return $found.ToArray()
}

$targets = New-Object 'System.Collections.Generic.List[object]'
foreach ($relative in @('build', '.tmp-worker-debug-2')) {
    $path = Join-Path $repoRoot $relative
    if (Test-Path -LiteralPath $path -PathType Container) {
        $targets.Add([pscustomobject]@{ Path = (Assert-ContainedPath $path); Kind = 'Directory' })
    }
}
$obsoleteExe = Join-Path $repoRoot 'dist\ChineseSemanticToolkit.exe'
if (Test-Path -LiteralPath $obsoleteExe -PathType Leaf) {
    $targets.Add([pscustomobject]@{ Path = (Assert-ContainedPath $obsoleteExe); Kind = 'File' })
}
foreach ($name in @('ADVANCED_PARAMETER_PLAN.md', 'ADVANCED_PARAMETER_VALIDATION.md', 'IMPLEMENTATION_PLAN.md', 'PARAMETER_PLAN.md', 'PARAMETER_VALIDATION.md')) {
    $path = Join-Path $repoRoot $name
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $targets.Add([pscustomobject]@{ Path = (Assert-ContainedPath $path); Kind = 'File' })
    }
}
foreach ($relative in @('assets', 'docs', 'examples', 'scripts', 'semantic_toolkit', 'tests')) {
    $scope = Join-Path $repoRoot $relative
    if (Test-Path -LiteralPath $scope -PathType Container) {
        foreach ($cache in (Find-CacheDirectories $scope)) {
            $targets.Add([pscustomobject]@{ Path = (Assert-ContainedPath $cache); Kind = 'Directory' })
        }
    }
}
$rootCache = Join-Path $repoRoot '__pycache__'
if (Test-Path -LiteralPath $rootCache -PathType Container) {
    $targets.Add([pscustomobject]@{ Path = (Assert-ContainedPath $rootCache); Kind = 'Directory' })
}

$unique = New-Object 'System.Collections.Generic.List[object]'
$seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
foreach ($target in $targets) {
    if ($seen.Add($target.Path)) { $unique.Add($target) }
}
$targets = $unique

Write-Host 'PYsemantic 源代码目录清理'
Write-Host '范围仅包括旧 build 输出、指定临时文件、旧版可执行文件、历史计划文件和 Python 缓存。'
Write-Host '最终发布文件 dist\PYsemantic.exe、ZIP、PDF、校验清单，以及 .venv、模型和测试均会保留。'
Write-Host ''
if ($targets.Count -eq 0) {
    Write-Host '没有找到需要清理的旧文件。'
    exit 0
}

foreach ($target in $targets) {
    if ($target.Kind -eq 'Directory') {
        $stats = Get-DirectoryStats $target.Path
        $relative = $target.Path.Substring($repoRoot.Length + 1)
        Write-Host ('  {0}\  {1:N0} bytes, {2:N0} files' -f $relative, $stats.Bytes, $stats.Files)
    } else {
        Assert-NoReparsePoints $target.Path
        $item = Get-Item -LiteralPath $target.Path -Force
        $relative = $target.Path.Substring($repoRoot.Length + 1)
        Write-Host ('  {0}  {1:N0} bytes' -f $relative, $item.Length)
    }
}

if (-not $Apply) {
    Write-Host ''
    Write-Host '当前为预览模式，没有删除文件。若要清理，请双击 cleanup-legacy.cmd。'
    exit 0
}

Write-Host ''
$answer = Read-Host '请确认 PYsemantic 已关闭；输入 Y 才会删除以上列出的内容'
if ($answer -cne 'Y') {
    Write-Host '已取消，没有删除文件。'
    exit 0
}

foreach ($target in $targets) {
    Assert-ContainedPath $target.Path | Out-Null
    if ($target.Kind -eq 'Directory') {
        Assert-NoReparsePoints $target.Path
        Remove-Item -LiteralPath $target.Path -Recurse -Force
    } else {
        Assert-NoReparsePoints $target.Path
        Remove-Item -LiteralPath $target.Path -Force
    }
}
Write-Host '清理完成。'
