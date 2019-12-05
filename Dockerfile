FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
LABEL name="haq-quantization-sampling"

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         vim \
         cmake \
         git \
	 python3-pip \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision

RUN git clone https://github.com/phrb/haq-release.git
RUN cd haq-release && git checkout quantization-sampling && cd ..

ENV http_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
ENV https_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
