#!/bin/bash
# -----------------------------------------------------------------------------
# Builds OpenCV with CUDA support and installs it in /usr/local
# -----------------------------------------------------------------------------

OPENCV_VERSION=4.12.0

# Starting with cuDNN v9, the header naming convention has changed. OpenCV (as of 4.12.0) still
# expects the original naming convention. So, create symlinks so that OpenCV cmake can find the
# necessary headers.
ARCH=$(uname -m); \
  [ ! -e /usr/include/${ARCH}-linux-gnu/cudnn.h ] && \
  sudo ln -s /usr/include/${ARCH}-linux-gnu/cudnn_v9.h /usr/include/${ARCH}-linux-gnu/cudnn.h || true; \
  [ ! -e /usr/include/${ARCH}-linux-gnu/cudnn_version.h ] && \
  sudo ln -s /usr/include/${ARCH}-linux-gnu/cudnn_version_v9.h /usr/include/${ARCH}-linux-gnu/cudnn_version.h || true;

mkdir /tmp/opencv
cd /tmp/opencv
test -e ${OPENCV_VERSION}.zip || wget https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip
test -e opencv-${OPENCV_VERSION} || unzip ${OPENCV_VERSION}.zip
test -e opencv_extra_${OPENCV_VERSION}.zip || wget -O opencv_extra_${OPENCV_VERSION}.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip
test -e opencv_contrib-${OPENCV_VERSION} || unzip opencv_extra_${OPENCV_VERSION}.zip

cd opencv-${OPENCV_VERSION}
mkdir build
cd build
cmake -G Ninja \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_SHARED_LIBS=ON \
      -D BUILD_LIST=core,imgproc,imgcodecs,videoio,video,calib3d,features2d,dnn,highgui,ximgproc,cudaarithm,cudaimgproc,cudawarping,cudev \
      -D WITH_TBB=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=OFF \
      -D WITH_OPENCL=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_DOCS=OFF \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_sfm=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D CUDNN_INCLUDE_DIR=/usr/include/$(uname -m)-linux-gnu \
      -D CUDNN_LIBRARY=/usr/lib/$(uname -m)-linux-gnu/libcudnn.so \
      ..

ninja -j"$(nproc)" && sudo ninja install && sudo ldconfig
rm -rf /tmp/opencv
