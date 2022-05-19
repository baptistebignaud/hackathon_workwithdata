ARG UBUNTU_VERSION=20.04
ARG CUDA=11.2
FROM nvcr.io/nvidia/tensorflow:21.08-tf2-py3
# FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-devel-ubuntu${UBUNTU_VERSION} as base
# CMD nvidia-smi
ARG PYTHON_VERSION=3.8

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    python3.8-dev\
    unixodbc-dev\
    build-essential\
    python-sqlalchemy\
    graphviz\
    ffmpeg libsm6 libxext6\
    # Change default python
    && cd /usr/bin \
    && ln -sf python${PYTHON_VERSION}         python3 \
    && ln -sf python${PYTHON_VERSION}m        python3m \
    && ln -sf python${PYTHON_VERSION}-config  python3-config \
    && ln -sf python${PYTHON_VERSION}m-config python3m-config \
    && ln -sf python3                         /usr/bin/python \
    # Update pip and add common packages
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade \
    setuptools \
    wheel \
    six \
    # Cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable

# install chromedriver
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# set display port to avoid crash
ENV DISPLAY=:99
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
WORKDIR /appli

