# MED-CXX

![Demo: Blood Vessel Segmentation with UNet](assets/demo_blood_vessels_seg_unet.gif)

MED-CXX is a C++ project built with an object-oriented design, aiming to solve key medical imaging problems through deep learning. The project focuses on implementing and training several convolutional neural network (CNN) architectures — including U-Net, ResNet, and DenseNet — entirely in C++ using LibTorch (the PyTorch C++ API).

These models are applied to core medical tasks such as blood vessel segmentation, disease detection from medical scans, and image-based classification of pathological conditions.

MED-CXX was developed as part of the Object-Oriented Programming (OOP) lab course project at the Faculty of Mathematics and Computer Science, University of Bucharest, with a focus on clarity, reusability, and clean class structure.

## Changelog

### Implemented Features (May 2025)

- **UNet**, **ResNet**, and **DenseNet** implemented using LibTorch
- **Blood vessel segmentation** on DRIVE/STARE datasets with visualization
- **Weighted BCE + Dice loss** for handling class imbalance
- **Model saving/loading** utilities (`saveModel`, `loadModel`)
- **GIF/video demo generator** for showing side-by-side input / ground truth / prediction
- **ImageLoader** with caching, preprocessing, and grayscale/threshold conversion
- Modular class structure (`BaseModel`, `ImageLoader`, `Benchmark`, etc.)
- Metrics: Accuracy, Precision, Recall, F1 Score, etc.
- Auto device selection (`CUDA` or `CPU`)

## Installation

### Linux

```bash
# Install dependencies
sudo apt update
sudo apt install cmake libopencv-dev build-essential

# Clone the repo
git clone https://github.com/IAMSebyi/med-cxx.git
cd med-cxx

# Download LibTorch (instead of <COMPUTE PLATFORM>, use cpu, cu118, cu126, cu128)
wget https://download.pytorch.org/libtorch/<COMPUTE PLATFORM>/libtorch-cxx11-abi-shared-with-deps-2.7.0%2B<COMPUTE PLATFORM>.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0%2B<COMPUTE PLATFORM>.zip

# Build the project
mkdir build && cd build
cmake ..
make
```

### Windows

#### 1. Install dependencies

- [CMake](https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-windows-x86_64.msi)
- [OpenCV](https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe)
- [Visual Studio 2022](https://visualstudio.microsoft.com/vs/)

#### 2. Clone the project

```bash
git clone https://github.com/IAMSebyi/med-cxx.git
cd med-cxx
```

#### 3. Download LibTorch (CPU or CUDA 11.8, 12.6, 12.8)

- [LibTorch](https://pytorch.org/)

#### 4. Configure and build using CMake

```bash
mkdir build && cd build
cmake ..
make
```

#### 5. Ensure *.dll files from libtorch/lib and opencv/build/x64/vcXX/bin are either in your PATH or next to your executable.

### Mac

```bash
# Install dependencies
brew install cmake opencv

# Clone the repo
git clone https://github.com/IAMSebyi/med-cxx.git
cd med-cxx

# Download LibTorch (only CPU supported)
https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
unzip libtorch-macos-arm64-2.7.0.zip

mkdir build && cd build
cmake ..
make
```

## Resources

- [**U-Net: Convolutional Networks for Biomedical Image Segmentation**](https://arxiv.org/abs/1505.04597) - Olaf Ronneberger, Philipp Fischer, Thomas Brox
- [**Deep Residual Learning for Image Recognition**](https://arxiv.org/abs/1512.03385) - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- [**Densely Connected Convolutional Networks**](https://arxiv.org/abs/1608.06993) - Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
