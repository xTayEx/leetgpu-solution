cmake_minimum_required(VERSION 3.10)
project("leetgpu" LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(vector_add vector_add.cu)
target_include_directories(vector_add PRIVATE include)
set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES "89")

add_executable(matmul matmul.cu)
target_include_directories(matmul PRIVATE include)
target_include_directories(matmul PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(matmul PRIVATE cublas)
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "89")
