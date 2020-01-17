FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
LABEL name="haq-quantization-sampling"

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential                                         \
         vim                                                     \
         cmake                                                   \
         git                                                     \
	 python3-pip                                             \
         python3-setuptools                                      \
         python3-pandas                                          \
         curl                                                    \
         ca-certificates                                         \
         libjpeg-dev                                             \
         libpng-dev                                              \
         liblapack-dev                                           \
         libopenblas-dev                                         \
         r-base &&                                               \
     rm -rf /var/lib/apt/lists/*

RUN Rscript -e 'install.packages(c("rsm", "dplyr", "DiceKriging", "DiceDesign", "DiceOptim","randtoolbox"), repos="https://cran.rstudio.com")'

RUN pip3 install               \
        'torch>=1.1'           \
        'torchvision>=0.3.0'   \
        'numpy>=1.14'          \
        'easydict>=1.8'        \
        'progress>=1.4'        \
        'matplotlib<3.1'       \
        'scikit-learn>=0.21.0' \
        'tensorboardX>=1.7'

ENV http_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
ENV https_proxy "http://web-proxy-pa.labs.hpecorp.net:8088/"
