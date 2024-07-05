cmake_minimum_required(VERSION 3.5.1)
project(kbd)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE sources ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
file(GLOB_RECURSE headers ${CMAKE_CURRENT_LIST_DIR}/src/*.h)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(ArrowDataset REQUIRED)

add_executable(${PROJECT_NAME} 
        main.cpp
        ${sources}
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
        ${ARROW_INCLUDE_DIR}
)

target_link_libraries(
        ${PROJECT_NAME}
        PRIVATE
        nlohmann_json::nlohmann_json
        Arrow::arrow_shared
)

