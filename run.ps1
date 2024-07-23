# Set script parameters and default values
param(
    [string]$rootDir = "D:\william\data\KBD",
    [switch]$multiple,
    [switch]$useGlobal
)

# Log file path
$logFile = "$rootDir\logfile.log"

# Function to append log messages with timestamps to a log file
function Write-Log {
    Param ([string]$message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $message" | Out-File -FilePath $logFile -Append
    "$timestamp - $message"
}

# Check if the root directory exists
if (-Not (Test-Path $rootDir)) {
    Write-Log "$rootDir does not exist"
    exit
}

# Processing directories
function Process-Directory {
    param([string]$basePath)
    Write-Log "Processing directory $basePath with global flag: $useGlobal"

    $csvPath = Get-ChildItem -Path $basePath -Filter "depthquality*.csv" | Select-Object -First 1 -ExpandProperty FullName
    if (-Not $csvPath) {
        $xlsxPath = Get-ChildItem -Path $basePath -Filter "depthquality*.xlsx" | Select-Object -First 1 -ExpandProperty FullName
        if ($xlsxPath) {
            $csvPath = "$xlsxPath".Replace('.xlsx', '.csv')
            python xlsx_to_csv.py $xlsxPath
            Write-Log "Converted XLSX to CSV: $csvPath"
        }
    } else {
        Write-Log "CSV found: $csvPath"
    }

    # Run .\kbd command
    $file_path = Join-Path $basePath "image_data"
    $output_path = Join-Path $basePath "image_data_lc++"
    $cmdArgs = "-f `"$file_path`" -c `"$csvPath`" -t `"$output_path`""
    if ($useGlobal) {
        $cmdArgs += " -g"
    }
   
    Set-Location -Path $PSScriptRoot

    # Construct the path to the executable
    $binary_path = Join-Path -Path $PSScriptRoot -ChildPath "build\Release"
    $executable_path = Join-Path -Path $binary_path -ChildPath "kbd.exe"

    # Output the path for verification
    Write-Host "Executable path: $executable_path"
    Write-Host "cmdArgs: $cmdArgs"

    # Check if the executable exists
    if (Test-Path $executable_path) {
        & $executable_path $cmdArgs | Out-File -FilePath $logFile -Append
    } else {
        Write-Host "Executable not found at $executable_path"
    }
    $cmdArgsArray = $cmdArgs -split ' '

    & $executable_path @cmdArgsArray | Out-File -FilePath $logFile -Append
}

# Determine mode and process accordingly
if ($multiple) {
    Get-ChildItem -Path $rootDir -Directory | ForEach-Object {
        Process-Directory -basePath $_.FullName
    }
} else {
    $specificDir = Join-Path $rootDir "N09ASH24DH0050"
    if (Test-Path $specificDir) {
        Process-Directory -basePath $specificDir
    } else {
        Write-Log "$specificDir does not exist"
    }
}
