cmake_minimum_required(VERSION 3.10)
project(Assignment2)

set(CMAKE_CXX_STANDARD 17)

# Eigen3
set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/../share/eigen3/cmake")
find_package(Eigen3 REQUIRED)

# OpenCV (MinGW 自编译)
set(OpenCV_DIR "Z:/Opencv/mingw-install")
find_package(OpenCV REQUIRED)

# 源文件
add_executable(Assignment2
    main.cpp
    rasterizer.hpp rasterizer.cpp
    global.hpp
    Triangle.hpp Triangle.cpp
)

# 链接
target_link_libraries(Assignment2 PRIVATE Eigen3::Eigen ${OpenCV_LIBS})

# rasterizer.hpp / Triangle.hpp 用了 <eigen3/Eigen/Eigen>，需要补一级 include
get_filename_component(WORKSPACE_ROOT "${Eigen3_DIR}/../../.." ABSOLUTE)
target_include_directories(Assignment2 PRIVATE "${WORKSPACE_ROOT}/include")