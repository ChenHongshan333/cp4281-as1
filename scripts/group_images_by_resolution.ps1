[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
    throw "Source image directory does not exist: $SourceDirectory"
}

$sourcePath = (Resolve-Path -LiteralPath $SourceDirectory).Path
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
if ($sourcePath -eq $outputPath) {
    throw "OutputDirectory must be different from SourceDirectory."
}

if ((Test-Path -LiteralPath $outputPath) -and
    (Get-ChildItem -LiteralPath $outputPath -Force | Select-Object -First 1)) {
    throw "Output directory is not empty: $outputPath"
}

$images = @(
    Get-ChildItem -LiteralPath $sourcePath -File |
        Where-Object { $_.Extension -match '^\.(jpg|jpeg)$' } |
        Sort-Object Name
)
if ($images.Count -eq 0) {
    throw "No JPG images found in: $sourcePath"
}

Add-Type -AssemblyName System.Drawing
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$groups = @{}

foreach ($file in $images) {
    $bitmap = [Drawing.Image]::FromFile($file.FullName)
    try {
        $groupName = "{0}x{1}" -f $bitmap.Width, $bitmap.Height
    }
    finally {
        $bitmap.Dispose()
    }

    $groupDirectory = Join-Path $outputPath $groupName
    New-Item -ItemType Directory -Force -Path $groupDirectory | Out-Null
    $destination = Join-Path $groupDirectory $file.Name
    New-Item -ItemType HardLink -Path $destination -Target $file.FullName |
        Out-Null

    if (-not $groups.ContainsKey($groupName)) {
        $groups[$groupName] = 0
    }
    $groups[$groupName] += 1
}

Write-Host "Grouped $($images.Count) images using space-efficient hard links:"
$groups.GetEnumerator() | Sort-Object Name | ForEach-Object {
    Write-Host "  $($_.Name): $($_.Value) images"
}
Write-Host "Output: $outputPath"
