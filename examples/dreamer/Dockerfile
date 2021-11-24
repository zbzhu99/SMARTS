FROM tensorflow/tensorflow:2.4.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# (Optional) Disable certificate verification
RUN touch /etc/apt/apt.conf.d/99verify-peer.conf && \
    echo >>/etc/apt/apt.conf.d/99verify-peer.conf 'Acquire::https::Verify-Peer "false";' && \
    echo >>/etc/apt/apt.conf.d/99verify-peer.conf 'Acquire::https::Verify-Host "false";'

# (Optional) Disable Ubuntu -security repositories
# RUN find /etc/apt -type f -name '*.list' -exec sed -i 's/\(^deb.*-backports.*\)/#\1/; s/\(^deb.*-updates.*\)/#\1/; s/\(^deb.*-proposed.*\)/#\1/; s/\(^deb.*-security.*\)/#\1/' {} +

# Install libraries
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update && \
    apt-get install -y \
        libspatialindex-dev \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        sumo \
        sumo-doc \
        sumo-tools \
        wget \
        xorg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Setup SUMO
ENV SUMO_HOME /usr/share/sumo

# Update default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Setup virtual environment and install pip
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip config set global.trusted-host "pypi.python.org files.pythonhosted.org pypi.org"
RUN pip install --upgrade pip

# Install requirements.txt
ARG REQUIREMENTS
COPY ${REQUIREMENTS} /tmp/requirements.txt
RUN pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --no-cache-dir -r /tmp/requirements.txt

# Copy source files and install SMARTS
COPY . /src
WORKDIR /src
RUN pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --no-cache-dir -e . && \
    cp -r /src/smarts.egg-info /media/smarts.egg-info

# Build scenarios and maps
RUN scl scenario build-all --clean ./scenarios/loop ./scenarios/cloverleaf ./scenarios/figure_eight ./scenarios/game_of_tag_demo_map

# For Envision
EXPOSE 8081

# Tensorflow: Optimizing compiler for machine learning
# https://www.tensorflow.org/xla
ENV TF_XLA_FLAGS --tf_xla_auto_jit=2

# Suppress message of missing /dev/input folder and copy smarts.egg-info if not there
RUN echo "mkdir -p /dev/input\n" \
         "if [[ ! -d /src/smarts.egg-info ]]; then" \
         "   cp -r /media/smarts.egg-info /src/smarts.egg-info;" \
         "   chmod -R 777 /src/smarts.egg-info;" \
         "fi" >> ~/.bashrc

SHELL ["/bin/bash", "-c", "-l"]