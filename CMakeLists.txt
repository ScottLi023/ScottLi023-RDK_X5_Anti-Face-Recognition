cmake_minimum_required(VERSION 3.5)

project(face_liveness_project)

# ---------- 依赖路径配置 ----------
if(NOT DEPS_ROOT)
    set(DEPS_ROOT "/open_explorer/samples/ai_toolchain/horizon_runtime_sample/code/deps_gcc11.3/x86")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# DNN Header path
include_directories("/usr/include")
# link_directories("/usr/lib")

# ---------- 编译选项 ----------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")

# ---------- 编译动态库 ----------
add_library(face_liveness SHARED face_liveness.cpp)

# 链接库
target_link_libraries(face_liveness
    ${OpenCV_LIBS}
    dnn
    # hbdk_sim_x86
    pthread
    rt
    dl
)

# ---------- 可执行文件 (用于测试) ----------
add_executable(main main.cc)
target_link_libraries(main face_liveness)