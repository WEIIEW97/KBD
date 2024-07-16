@echo off
setlocal

if "%1"=="" (
    echo Usage: %0 [debug|release]
    exit /b 1
)

set BUILD_TYPE=%1

if "%BUILD_TYPE%"=="debug" (
    set CMAKE_BUILD_TYPE=Debug
) else if "%BUILD_TYPE%"=="release" (
    set CMAKE_BUILD_TYPE=Release
) else (
    echo Invalid build type: %BUILD_TYPE%
    echo Usage: %0 [debug|release]
    exit /b 1
)

rem Remove previous build files
if exist build\CMakeCache.txt (
    del /f /q build\CMakeCache.txt
)
if exist build\CMakeFiles (
    rmdir /s /q build\CMakeFiles
)

rem Create build directory if it doesn't exist
if not exist build (
    mkdir build
)

rem Navigate to the build directory
cd build

rem Run CMake with the specified build type and OpenMP support
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/lib/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE% -DCMAKE_CXX_FLAGS_RELEASE="/O2 /openmp" ..

rem Build the project with MSVC
cmake --build . --config %CMAKE_BUILD_TYPE%

rem Run the compiled program
if "%BUILD_TYPE%"=="debug" (
    Debug\kbd.exe
) else (
    Release\kbd.exe
)

endlocal
