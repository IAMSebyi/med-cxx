cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(med-cxx LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------------------------------------------------------------------
# 3) LibTorch (folderul libtorch/ în rădăcina proiectului)
# -------------------------------------------------------------------
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Found LibTorch ${TORCH_VERSION}")

# -------------------------------------------------------------------
# 4) OpenCV (din libopencv-dev)
# -------------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV ${OpenCV_VERSION}")

# -------------------------------------------------------------------
# 5) Executabil
# -------------------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/common/Loss.cpp
    src/data/ImageLoader.cpp
    src/evaluation/Benchmark.cpp
    src/models/BaseModel.cpp
    src/models/DenseNet.cpp
    src/models/ResNet.cpp  
    src/models/UNet.cpp 
    src/main.cpp
)

# 5.1) Include-uri
target_include_directories(${PROJECT_NAME}
    PRIVATE
      ${OpenCV_INCLUDE_DIRS}
      ${CUDA_INCLUDE_DIRS}
)

# 5.2) Link-uire
target_link_libraries(${PROJECT_NAME}
    PRIVATE
      ${TORCH_LIBRARIES}      # PyTorch C++ API
      ${OpenCV_LIBS}          # OpenCV
      ${CUDA_CUDART_LIBRARY}  # runtime-ul CUDA
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

# ----------------------------------------------------------------------------
# 6) (Opțional) Instalare
# ----------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
