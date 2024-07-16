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
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
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
        ${fmt_INCLUDE_DIRS}
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${ARROW_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        nlohmann_json::nlohmann_json
        Arrow::arrow_shared
        fmt::fmt
        ${CERES_LIBRARIES}
        OpenMP::OpenMP_CXX
)

target_compile_definitions(
        ${PROJECT_NAME}
        PRIVATE
        EIGEN_USE_THREADS
        # EIGEN_USE_MKL_ALL       
)
