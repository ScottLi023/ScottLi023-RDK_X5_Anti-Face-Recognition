cmake_minimum_required(VERSION 3.5)
project(face_liveness_v2)

# ---------- Dependencies ----------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Horizon DNN SDK headers
include_directories("/usr/include")

# ---------- Compiler Options ----------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")

# ---------- Shared Library (for Python ctypes) ----------
add_library(face_liveness SHARED face_liveness.cpp)
target_link_libraries(face_liveness
    ${OpenCV_LIBS}
    dnn
    pthread
    rt
    dl
)

# ---------- Test Executable ----------
add_executable(main main.cc)
target_link_libraries(main face_liveness)
