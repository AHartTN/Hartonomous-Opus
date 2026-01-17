Clear-Host

# Force UTF-8 encoding for all output
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$root = "D:\Repositories\Hartonomous-Opus"
$build = "$root\cpp\build"
$logs  = "$root\logs"
$proc  = [Environment]::ProcessorCount

Set-Location $root

Remove-Item -Path $build -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Path $build -Force | Out-Null
New-Item -ItemType Directory -Path $logs  -Force | Out-Null

Write-Host "=== Step 1: CMake Configure ===" -ForegroundColor Cyan
cmake -B $build -S "$root\cpp" 2>&1 | Out-File -FilePath "$logs\01_configure.txt" -Encoding utf8

Write-Host "=== Step 2: Build (Release) ===" -ForegroundColor Cyan
cmake --build $build --config Release -j $proc 2>&1 | Out-File -FilePath "$logs\02_compile.txt" -Encoding utf8

Write-Host "=== Step 3: Run Tests ===" -ForegroundColor Cyan
Push-Location $build
ctest -V -C Release -j $proc -T test -R ".*" 2>&1 | Out-File -FilePath "$logs\03_testcpp.txt" -Encoding utf8
Pop-Location

Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host "Logs written to: $logs"
