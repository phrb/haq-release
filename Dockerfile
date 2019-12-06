FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
LABEL name="haq-quantization-sampling"

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         vim \
         cmake \
         git \
	 python3-pip \
         python3-setuptools \
         python3-pandas \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        torch \
        torchvision \
        numpy \
        easydict \
        progress \
        'matplotlib<3.1' \
        scikit-learn \
        torch \
        torchvision \
        tensorboardX

ENV http_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
ENV https_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
