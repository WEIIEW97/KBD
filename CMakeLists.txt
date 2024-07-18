cmake_minimum_required(VERSION 3.5.1)
project(kbd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB_RECURSE headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Arrow REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# Platform-specific settings
if(MSVC)
    # Enable OpenMP for MSVC
    find_package(OpenMP REQUIRED)
    add_compile_options(/openmp)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
elseif(APPLE)
    # Disable OpenMP for macOS
    message(STATUS "OpenMP is disabled for macOS.")
else()
    # Enable OpenMP for Linux and other Unix-like systems
    find_package(Threads REQUIRED)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()


add_executable(${PROJECT_NAME} 
        # main.cpp
        test/test_main.cpp
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${Boost_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${ARROW_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

if(OpenMP_CXX_FOUND AND NOT APPLE)
        add_definitions(-DUSE_OPENMP)
        target_link_libraries(
                ${PROJECT_NAME}
                PRIVATE
                nlohmann_json::nlohmann_json
                Arrow::arrow_shared
                fmt::fmt
                ${CERES_LIBRARIES}
                ${Boost_PROGRAM_OPTIONS_LIBRARY}
                OpenMP::OpenMP_CXX
        )
else()
        target_link_libraries(
                ${PROJECT_NAME}
                PRIVATE
                nlohmann_json::nlohmann_json
                Arrow::arrow_shared
                fmt::fmt
                ${CERES_LIBRARIES}
                ${Boost_PROGRAM_OPTIONS_LIBRARY}
        )  
        endif()


target_compile_definitions(
        ${PROJECT_NAME}
        PRIVATE
        EIGEN_USE_THREADS
        # EIGEN_USE_MKL_ALL       
)
