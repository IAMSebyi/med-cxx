cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(med-cxx LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Found LibTorch ${TORCH_VERSION}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

# Executable
add_executable(${PROJECT_NAME}
    src/common/ArgParser.cpp 
    src/common/Exception.cpp
    src/common/Loss.cpp
    src/common/Utils.cpp
    src/common/Visualizer.cpp
    src/data/ImageLoader.cpp
    src/evaluation/Benchmark.cpp
    src/layers/BaseLayer.cpp
    src/layers/DenseLayer.cpp 
    src/layers/DenseBlock.cpp 
    src/layers/Transition.cpp 
    src/layers/BasicBlock.cpp 
    src/layers/Bottleneck.cpp
    src/layers/DoubleConv.cpp 
    src/layers/Down.cpp 
    src/layers/Up.cpp 
    src/layers/OutConv.cpp 
    src/models/BaseModel.cpp
    src/models/DenseNet.cpp
    src/models/ResNet.cpp  
    src/models/UNet.cpp 
    src/trainer/BaseTrainer.cpp
    src/trainer/ClassificationTrainer.cpp
    src/trainer/SegmentationTrainer.cpp
    src/runners/main.cpp
)

# Includes
target_include_directories(${PROJECT_NAME}
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src 
      ${OpenCV_INCLUDE_DIRS}
      ${CUDA_INCLUDE_DIRS}
)

# Linker flags
target_link_libraries(${PROJECT_NAME}
    PRIVATE
      ${TORCH_LIBRARIES}      # PyTorch C++ API
      ${OpenCV_LIBS}          # OpenCV
      ${CUDA_CUDART_LIBRARY}  # runtime-ul CUDA
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
