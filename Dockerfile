FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 AS compile-image

# Install a recent Python version
RUN apt-get update \
    && apt-get upgrade -qq \
    && apt-get install -qq \
        libpython3-dev \
        libqt5opengl5-dev \
        libqt5svg5-dev \
        pybind11-dev \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        qttools5-dev \
        qttools5-dev-tools \
        qt5-default

# Create venv and prefix to path
ENV VIRTUAL_ENV=/opt/venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.8 \
    && python3.8 -m pip install -r requirements.txt

# Build and install CloudCompare plugin
RUN git clone https://github.com/tmontaigu/CloudCompare-PythonPlugin.git /tmp/CloudCompare-PythonPlugin \
    && python3.8 -m pip install --use-feature=in-tree-build /tmp/CloudCompare-PythonPlugin/wrapper/pycc

# # Continue from fresh image
# FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04 AS build-image

# # Continue as unprivileged user
# RUN useradd --create-home appuser
# WORKDIR /home/appuser
# USER appuser
# COPY --from=compile-image --chown=appuser /opt/venv /opt/venv

# # Reset the path
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
