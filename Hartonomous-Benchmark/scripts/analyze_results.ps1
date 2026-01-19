param(
    [switch]$Detailed,
    [string]$Format = "summary",
    [string]$Csv,
    [string]$Json,
    [string]$Compare,
    [switch]$Verbose,
    [switch]$Help
)

# Starting Directory (Where you were when you ran the script)
$StartDir = Get-Location
# Default values
$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
$ResultsDir = "results"
$ScriptName = $MyInvocation.MyCommand.Name

# Function to display usage
function Show-Usage {
    $usage = @"
Usage: $ScriptName [OPTIONS]

Analyze benchmark results with various output formats and comparisons.

Options:
  -Detailed              Show detailed analysis including statistics
  -Format FORMAT         Output format: summary, detailed, csv, json (default: summary)
  -Csv FILE              Export results to CSV file
  -Json FILE             Export results to JSON file
  -Compare FILE          Compare with previous results file
  -Verbose               Enable verbose output
  -Help                  Display this help message

Examples:
  $ScriptName -Detailed
  $ScriptName -Format csv -Csv analysis.csv
  $ScriptName -Compare results/previous.json
"@
    Write-Host $usage
    exit 1
}

if ($Help) {
    Show-Usage
}

# Check if results directory exists
if (!(Test-Path $ResultsDir)) {
    Write-Host "Error: Results directory '$ResultsDir' not found."
    Write-Host "Run .\scripts\run_benchmarks.ps1 first."
    exit 1
}

# Start logging to file
Start-Transcript -Path "..\logs\analyze.txt" -Append

# Function to calculate statistics
function Calculate-Stats {
    param([string]$File)
    try {
        $data = Get-Content $File | ConvertFrom-Json
        $benchmarks = $data.benchmarks
    } catch {
        Write-Host "Error: Failed to parse JSON file $File. $($_.Exception.Message)"
        return
    }

    Write-Host "=== Benchmark Statistics ==="
    Write-Host "Total benchmarks: $($benchmarks.Count)"

    if ($benchmarks.Count -gt 0) {
        $avgTime = ($benchmarks | Measure-Object -Property real_time -Average).Average
        Write-Host ("Average real time: {0:F2} ns" -f $avgTime)

        $sortedTimes = $benchmarks | Sort-Object real_time
        $medianTime = $sortedTimes[($sortedTimes.Count / 2)].real_time
        Write-Host ("Median real time: {0:F2} ns" -f $medianTime)

        $variance = ($benchmarks | ForEach-Object { [math]::Pow($_.real_time - $avgTime, 2) } | Measure-Object -Sum).Sum / $benchmarks.Count
        $stdDev = [math]::Sqrt($variance)
        Write-Host ("Standard deviation: {0:F2} ns" -f $stdDev)

        $minTime = ($benchmarks | Measure-Object -Property real_time -Minimum).Minimum
        $maxTime = ($benchmarks | Measure-Object -Property real_time -Maximum).Maximum
        Write-Host ("Fastest benchmark: {0:F2} ns" -f $minTime)
        Write-Host ("Slowest benchmark: {0:F2} ns" -f $maxTime)
    }
}

# Function to generate CSV
function Generate-Csv {
    param([string]$InputFile, [string]$OutputFile)
    try {
        $data = Get-Content $InputFile | ConvertFrom-Json
        $csvData = $data.benchmarks | Select-Object name, real_time, cpu_time, time_unit, iterations
        $csvData | Export-Csv -Path $OutputFile -NoTypeInformation
        Write-Host "CSV exported to $OutputFile"
    } catch {
        Write-Host "Error: Failed to parse JSON file $InputFile for CSV export. $($_.Exception.Message)"
    }
}

# Function to compare results
function Compare-Results {
    param([string]$File1, [string]$File2)
    try {
        $data1 = Get-Content $File1 | ConvertFrom-Json
        $data2 = Get-Content $File2 | ConvertFrom-Json
    } catch {
        Write-Host "Error: Failed to parse JSON files for comparison. $($_.Exception.Message)"
        return
    }

    $names1 = $data1.benchmarks | Select-Object -ExpandProperty name
    $names2 = $data2.benchmarks | Select-Object -ExpandProperty name
    $commonNames = $names1 | Where-Object { $names2 -contains $_ }

    Write-Host "=== Results Comparison ==="
    foreach ($name in $commonNames) {
        $time1 = ($data1.benchmarks | Where-Object { $_.name -eq $name }).real_time
        $time2 = ($data2.benchmarks | Where-Object { $_.name -eq $name }).real_time
        if ($time1 -gt 0 -and $time2 -gt 0) {
            $change = (($time2 - $time1) / $time1) * 100
            Write-Host ("{0}: {1} -> {2} ({3}%)" -f $name, $time1, $time2, [math]::Floor($change))
        }
    }
}

# Main analysis
Write-Host "Benchmark Results Analysis"
Write-Host "=========================="

# Hardware info
$HardwareFile = Join-Path $ResultsDir "hardware_info.txt"
if (Test-Path $HardwareFile) {
    Write-Host ""
    Write-Host "Hardware Information:"
    Write-Host "---------------------"
    Get-Content $HardwareFile
}

# Perf stat
$PerfFile = Join-Path $ResultsDir "perf_stat.txt"
if (Test-Path $PerfFile) {
    Write-Host ""
    Write-Host "Performance Statistics:"
    Write-Host "-----------------------"
    Get-Content $PerfFile
}

# Results analysis
$ResultsFile = Join-Path $ResultsDir "results.json"
if (Test-Path $ResultsFile) {
    Write-Host ""
    Write-Host "Benchmark Results:"
    Write-Host "------------------"

    switch ($Format) {
        "summary" {
            Calculate-Stats -File $ResultsFile
            Write-Host ""
            Write-Host "Detailed Benchmark Results:"
            Write-Host "---------------------------"
            try {
                $data = Get-Content $ResultsFile | ConvertFrom-Json
                $table = $data.benchmarks | Select-Object @{Name="Benchmark Name"; Expression={$_.name}},
                                                          @{Name="Real Time"; Expression={$_.real_time}},
                                                          @{Name="CPU Time"; Expression={$_.cpu_time}},
                                                          @{Name="Time Unit"; Expression={$_.time_unit}},
                                                          @{Name="Iterations"; Expression={$_.iterations}}
                $table | Format-Table -AutoSize
            } catch {
                Write-Host "Error: Failed to parse results file. $($_.Exception.Message)"
            }
        }
        "detailed" {
            Calculate-Stats -File $ResultsFile
            Write-Host ""
            Write-Host "All Benchmark Results (Detailed):"
            Write-Host "----------------------------------"
            $data = Get-Content $ResultsFile | ConvertFrom-Json
            $table = $data.benchmarks | Select-Object @{Name="Benchmark Name"; Expression={$_.name}},
                                                      @{Name="Real Time"; Expression={$_.real_time}},
                                                      @{Name="CPU Time"; Expression={$_.cpu_time}},
                                                      @{Name="Time Unit"; Expression={$_.time_unit}},
                                                      @{Name="Iterations"; Expression={$_.iterations}},
                                                      @{Name="Items/sec"; Expression={if ($_.items_per_second) { $_.items_per_second } else { "N/A" }}}
            $table | Format-Table -AutoSize
        }
        "csv" {
            if ($Csv) {
                Generate-Csv -InputFile $ResultsFile -OutputFile $Csv
            } else {
                Generate-Csv -InputFile $ResultsFile -OutputFile (Join-Path $ResultsDir "analysis.csv")
            }
        }
        "json" {
            if ($Json) {
                Copy-Item $ResultsFile $Json
                Write-Host "JSON exported to $Json"
            } else {
                Write-Host "JSON results available at $ResultsFile"
            }
        }
    }

    # Comparison
    if ($Compare) {
        if (Test-Path $Compare) {
            Write-Host ""
            Compare-Results -File1 $ResultsFile -File2 $Compare
        } else {
            Write-Host "Warning: Comparison file '$Compare' not found."
        }
    }

} else {
    Write-Host "No benchmark results found. Run .\scripts\run_benchmarks.ps1 first."
}

# Additional exports
if ($Csv -and $Format -ne "csv") {
    Generate-Csv -InputFile $ResultsFile -OutputFile $Csv
}

if ($Json -and $Format -ne "json") {
    Copy-Item $ResultsFile $Json
    Write-Host "JSON exported to $Json"
}

Write-Host ""
Write-Host "Analysis complete."

# Stop logging
Stop-Transcript

Set-Location $StartDir