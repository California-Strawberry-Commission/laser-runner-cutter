ARG L4T_VERSION=r36.4.0
FROM nvcr.io/nvidia/l4t-jetpack:${L4T_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

WORKDIR /workspaces/ros2

# ROS 2 Humble
# Based on https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
RUN apt-get update && apt-get install -y software-properties-common curl python3-pip && \
    add-apt-repository universe -y && \
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb && \
    rm /tmp/ros2-apt-source.deb && \
    apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-image-transport-plugins \
    ros-dev-tools \
    ros-${ROS_DISTRO}-rosbridge-suite \
    ros-${ROS_DISTRO}-async-web-server-cpp \
    ros-${ROS_DISTRO}-diagnostic-updater \
    && rm -rf /var/lib/apt/lists/*

# GStreamer
# Based on https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c
RUN apt-get update && apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-nice \
    && rm -rf /var/lib/apt/lists/*

# OpenCV with CUDA support
COPY src/detection/scripts/build_opencv_cuda.sh /tmp/build_opencv_cuda.sh
RUN apt-get update && apt-get install -y wget unzip cmake ninja-build && \
    bash /tmp/build_opencv_cuda.sh && \
    rm -rf /var/lib/apt/lists/*

# Foundational Python packages
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    "setuptools==58.2.0" \
    wheel \
    ninja \
    packaging \
    pyproject_metadata \
    meson-python

# Arena SDK
COPY docker/ArenaSDK /tmp/arena
RUN ARCH=$(uname -m) && \
    echo "Installing Arena SDK for ${ARCH}" && \
    mkdir -p /opt/ArenaSDK && \
    if [ "${ARCH}" = "x86_64" ]; then \
    cp -a /tmp/arena/x64/. /opt/ArenaSDK/ArenaSDK_Linux_x64/ && \
    cd /opt/ArenaSDK/ArenaSDK_Linux_x64 && \
    sh Arena_SDK_Linux_x64.conf; \
    elif [ "${ARCH}" = "aarch64" ]; then \
    cp -a /tmp/arena/ARM64/. /opt/ArenaSDK/ArenaSDK_Linux_ARM64/ && \
    cd /opt/ArenaSDK/ArenaSDK_Linux_ARM64 && \
    sh Arena_SDK_ARM64.conf; \
    else \
    echo "Unsupported architecture: ${ARCH}"; exit 1; \
    fi && \
    python3 -m pip install --no-cache-dir /tmp/arena/arena_api-2.5.9-py3-none-any.whl && \
    rm -rf /tmp/arena

# RealSense
# Based on https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages
RUN apt-get update && apt-get install -y apt-transport-https curl && \
    mkdir -p /etc/apt/keyrings && \
    curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | \
    gpg --dearmor | tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo `lsb_release -cs` main" | \
    tee /etc/apt/sources.list.d/librealsense.list && \
    apt-get update && apt-get install -y \
    librealsense2-dev \
    librealsense2-dbg \
    ros-${ROS_DISTRO}-librealsense2* \
    && rm -rf /var/lib/apt/lists/*

# APT packages required by our ROS nodes
RUN apt-get update && apt-get install -y \
    libcairo2-dev \
    libsrtp2-1 \
    libgirepository1.0-dev \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages required by our ROS nodes
RUN python3 -m pip install --no-cache-dir \
    "numpy<2.0" \
    scipy \
    asttokens varname pure-eval \
    farm-ng-amiga \
    pyrealsense2 \
    "PyGObject==3.50.0" \
    livekit-api

# Install aioros2. Note that it is a local package and thus needs to be installed after src/
# has been copied.
COPY src/ src/
RUN pip install --no-cache-dir -e src/aioros2

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
