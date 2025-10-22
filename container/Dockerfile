ARG DISTRIBUTION_ROS=humble
ARG VERSION_PYTHON=3.10


# ========================================
FROM amd64/ros:"$DISTRIBUTION_ROS-perception" AS base



# ========================================
FROM base AS system

# Set shell
SHELL ["/bin/bash", "-c"]

# Configure shell
ARG DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS="yes"

# Install essentials
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    nano \
    software-properties-common \
    tmux \
    unzip \
    wget \
    x11-apps \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Enable GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install CUDA Toolkit
RUN wget --quiet "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" --directory-prefix /tmp/ && \
    dpkg -i /tmp/cuda-keyring_1.1-1_all.deb && \
    rm -rf /tmp/*
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    cuda-toolkit-12-4 \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/local/cuda/bin:$PATH"
RUN ldconfig

# Install cuDNN
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    cudnn9-cuda-12 \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install TensorRT
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    tensorrt \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
ARG DISTRIBUTION_ROS=humble
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    gettext-base \
    ros-dev-tools \
    "ros-$DISTRIBUTION_ROS-rmw-cyclonedds-cpp" \
    "ros-$DISTRIBUTION_ROS-moveit-msgs" \
    "ros-$DISTRIBUTION_ROS-vision-msgs" \
    "ros-$DISTRIBUTION_ROS-rosbag2" \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Add group and user
ARG USER=user \
    UID=1001
RUN addgroup --gid "$UID" "$USER" && \
    adduser --uid "$UID" --gid "$UID" --disabled-password "$USER" && \
    usermod -aG sudo "$USER" && \
    echo "$USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers



# ========================================
FROM system AS python

# Install Python
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Install Python version explicitly
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
    "python$VERSION_PYTHON" \
    "python$VERSION_PYTHON-venv" \
    && \
    apt-get autoremove -y -qq && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/*

# Configure Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore:::setuptools.command.install,ignore:::setuptools.command.easy_install,ignore:::setuptools.command.develop,ignore:::setuptools.dist,ignore:::pkg_resources \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Setup venv
RUN "python$VERSION_PYTHON" -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/venv/lib/python$VERSION_PYTHON/site-packages:$PYTHONPATH"
RUN pip install --no-cache-dir \
    pip==24.2 \
    setuptools==74.0.0 \
    wheel==0.44.0



# ========================================
FROM python AS libs

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    rm -rf /tmp/*



# ========================================
FROM libs AS person-detection-2d-lidar

# Add code
ARG USER=user \
    UID=1001
RUN git clone https://github.com/bertan-karacora/markers_server.git "/home/$USER/colcon_ws/src/markers_server" && \
    mkdir --parents "/home/$USER/data/maps"
RUN git clone https://github.com/bertan-karacora/ros2_interfaces.git "/home/$USER/colcon_ws/src/ros2_interfaces"
RUN git clone https://github.com/bertan-karacora/ros2_utils.git "/home/$USER/colcon_ws/src/ros2_utils"
COPY . "/home/$USER/colcon_ws/src/ros2_person_detection_2d_lidar"
RUN chown --recursive "$USER:$USER" "/home/$USER"

# Build
ARG DISTRIBUTION_ROS=humble
USER "$USER"
WORKDIR "/home/$USER/colcon_ws"
RUN source "/opt/ros/$DISTRIBUTION_ROS/setup.bash" && \
    colcon build --symlink-install

# Setup run
WORKDIR "/home/$USER/colcon_ws/src/ros2_person_detection_2d_lidar"
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["scripts/start_all.sh"]