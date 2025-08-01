VERSION=4.12.0

test -e ${VERSION}.zip || wget https://github.com/opencv/opencv/archive/refs/tags/${VERSION}.zip
test -e opencv-${VERSION} || unzip ${VERSION}.zip

test -e opencv_extra_${VERSION}.zip || wget -O opencv_extra_${VERSION}.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/${VERSION}.zip
test -e opencv_contrib-${VERSION} || unzip opencv_extra_${VERSION}.zip

# Starting with cuDNN v9, the header naming convention has changed. OpenCV (as of 4.12.0) still
# expects the original naming convention. So, create symlinks so that OpenCV cmake can find the
# necessary headers.
[ ! -e /usr/include/x86_64-linux-gnu/cudnn.h ] && sudo ln -s /usr/include/x86_64-linux-gnu/cudnn_v9.h /usr/include/x86_64-linux-gnu/cudnn.h
[ ! -e /usr/include/x86_64-linux-gnu/cudnn_version.h ] && sudo ln -s /usr/include/x86_64-linux-gnu/cudnn_version_v9.h /usr/include/x86_64-linux-gnu/cudnn_version.h

cd opencv-${VERSION}
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=OFF \
-D BUILD_opencv_apps=OFF \
-D BUILD_opencv_python2=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${VERSION}/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D WITH_FFMPEG=ON \
-D CUDNN_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
-D CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so \
..

make -j 8
sudo make -j 8 install
