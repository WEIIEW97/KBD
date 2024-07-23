@echo off
setlocal

:: Function to log messages with timestamps
:log
echo [%date% %time%] - %~1 >> "%log_file%"
goto :eof

:: Handling command line arguments
set "mode=single"
set "use_global=false"
set "root_dir=D:\william\data\KBD"
set "log_file=%root_dir%\logfile.log"

if not exist "%root_dir%" (
    call :log "%root_dir% does not exist"
    goto end_script
)

:parse_args
if "%1"=="" goto run
if "%1"=="-m" set "mode=multiple" & shift & goto parse_args
if "%1"=="-g" set "use_global=true" & shift & goto parse_args
call :log "Usage: %0 [-m for multiple directories] [-g to use global flag]"
exit /b 1

:run
if "%mode%"=="multiple" (
    call :log "Starting in multiple directory mode"
    if not exist "%root_dir%\*" (
        call :log "No directories found in the specified root directory"
        goto :eof
    )
    for /d %%d in ("%root_dir%\*") do (
        call :log "Found directory: %%d"
        if exist "%%d\*" (
            call :log "Processing directory %%d with global flag: %use_global%"
            call :process_directory "%%d"
        ) else (
            call :log "Directory %%d is empty or inaccessible"
        )
    )
) else (
    set "specific_dir=%root_dir%\N09ASH24DH0050"
    call :log "Processing single directory %specific_dir% with global flag: %use_global%"
    call :process_directory "%specific_dir%"
)

goto :eof

:process_directory
set "base_path=%~1"
set "file_path=%base_path%\image_data"
set "output_path=%base_path%\image_data_lc++"

for /f "delims=" %%f in ('dir /b /a-d "%base_path%\depthquality*.csv"') do set "csv_path=%base_path%\%%f" & goto found_csv
set "csv_path="
:found_csv

if not defined csv_path (
    for /f "delims=" %%x in ('dir /b /a-d "%base_path%\depthquality*.xlsx"') do (
        set "xlsx_path=%base_path%\%%x"
        python xlsx_to_csv.py "%xlsx_path%"
        set "csv_path=%xlsx_path:.xlsx=.csv%"
        call :log "Converted XLSX to CSV: %csv_path%"
        goto process_files
    )
) else (
    call :log "CSV found: %csv_path%"
)

:process_files
cd build
if "%use_global%"=="true" (
    .\kbd -f "%file_path%" -c "%csv_path%" -t "%output_path%" -g >> "%log_file%" 2>&1
) else (
    .\kbd -f "%file_path%" -c "%csv_path%" -t "%output_path%" >> "%log_file%" 2>&1
)
cd ..

goto :eof
