FROM osrf/ros:kinetic-desktop
MAINTAINER Alexandros Philotheou alefilot@auth.gr
# ------------------------------------------------------------------------------
RUN apt-get update
RUN apt-get install -y \
      sudo \
      apt-utils \
      build-essential \
      g++ \
      git \
      libfftw3-dev \
      libcgal-dev=4.7-4 \
      libcgal-qt5-dev=4.7-4 \
      libboost-random-dev \
      curl \
      lsb-release \
      python-catkin-tools \
      python-rosdep \
      python-rosinstall \
      python-rosinstall-generator \
      python-wstool \
      build-essential \
      python-rosdep \
      ros-kinetic-csm \
      ros-kinetic-pcl-ros \
      ros-kinetic-map-server
# ------------------------------------------------------------------------------
# Use bash and create user
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
# ------------------------------------------------------------------------------
# Create new user and home directory
# https://roboticseabass.com/2023/07/09/updated-guide-docker-and-ros2/
ARG USER=cbgl
RUN useradd --create-home ${USER}                                           && \
    echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER}         && \
    chmod 0440 /etc/sudoers.d/${USER}                                       && \
    mkdir -p /home/${USER}
# ------------------------------------------------------------------------------
USER ${USER}
WORKDIR /home/${USER}
RUN echo "source /opt/ros/kinetic/setup.bash" >> /home/${USER}/.bashrc
RUN echo "source /home/${USER}/catkin_ws/devel/setup.bash" >> /home/${USER}/.bashrc
# ------------------------------------------------------------------------------
RUN rosdep update
RUN mkdir -p /home/${USER}/catkin_ws/src && \
    cd /home/${USER}/catkin_ws/src/
# ------------------------------------------------------------------------------
COPY --chown=${USER}:${USER} ../cbgl/ /home/${USER}/catkin_ws/src/cbgl/
# ------------------------------------------------------------------------------
RUN cd /home/${USER}/catkin_ws && \
    export CC=gcc && \
    export CXX=g++ && \
    alias g++='g++ -std=c++11' && \
    alias clang++='clang++ -std=c++11' && \
    source /opt/ros/kinetic/setup.bash && \
    catkin build cbgl && \
    source /opt/ros/kinetic/setup.bash && \
    source /home/${USER}/catkin_ws/devel/setup.bash
# ------------------------------------------------------------------------------
# The next five lines + the entrypoint command will make sure that
# when the container is run the cbgl node is roslaunched immediately
RUN echo "#!/bin/bash" > /home/${USER}/cbgl_launch.sh
RUN echo "source /opt/ros/kinetic/setup.bash" >> /home/${USER}/cbgl_launch.sh
RUN echo "source /home/${USER}/catkin_ws/devel/setup.bash" >> /home/${USER}/cbgl_launch.sh
RUN echo "roslaunch cbgl cbgl.launch" >> /home/${USER}/cbgl_launch.sh
RUN chmod +x /home/${USER}/cbgl_launch.sh
# ------------------------------------------------------------------------------
USER root
ENTRYPOINT bash /home/${USER}/cbgl_launch.sh
