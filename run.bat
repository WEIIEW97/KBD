@echo off
setlocal

:: Set default mode and global flag
set mode=single
set use_global=false
set root_dir=C:\Users\williamwei\Codes\KBD\data

:: Parse command line arguments for '-m' (multiple) and '-g' (global)
:argloop
if "%1"=="" goto argdone
if "%1"=="-m" set mode=multiple & shift & goto argloop
if "%1"=="-g" set use_global=true & shift & goto argloop
shift
goto argloop
:argdone

:: Function to process directories
call :process_directory specific_dir %use_global%
goto end

:process_directory
set base_path=%1
set global_flag=%2
set file_path=%base_path%\image_data
for /f "delims=" %%f in ('dir "%base_path%\depthquality-*.csv" /b /a:-d /o:d') do set csv_path=%base_path%\%%f & goto foundcsv
:foundcsv
set output_path=%base_path%\image_data_lc++

:: Navigate to the build directory and execute the program
cd build
if "%global_flag%"=="true" (
    .\kbd -f "%file_path%" -c "%csv_path%" -t "%output_path%" -g
) else (
    .\kbd -f "%file_path%" -c "%csv_path%" -t "%output_path%"
)
cd ..
goto :eof

:end
if "%mode%"=="multiple" (
    :: Process all directories
    for /d %%d in (%root_dir%\*) do (
        echo Processing directory %%d with global flag: %use_global%
        call :process_directory "%%d" %use_global%
    )
) else (
    :: Process only the specified directory
    set specific_dir=%root_dir%\N09ASH24DH0050
    echo Processing single directory %specific_dir% with global flag: %use_global%
    call :process_directory "%specific_dir%" %use_global%
)

:end_script
endlocal
