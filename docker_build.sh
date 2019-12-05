#! /bin/bash

set -e

function usage() {
    echo "Usage: ./docker_build.sh [OPTION]"
    echo "  -b, --build       Builds the docker image"
    echo "  -r, --run         Launches a container, gets a shell"
    echo "  -c, --configure   Configure repository for experiments"
    echo "  --help            Print this message"
    exit 0
}

if [ $# -eq 0 ] ; then
    usage
fi

PROXY_URL=http://web-proxy-pa.labs.hpecorp.net:8088/
SRC_DIR=/shared/bruelp/imagenet
TARGET_DIR=/haq-release/data

while test $# -gt 0
do
    case "$1" in
        -b|--build)
            sudo docker build \
                 -t haq-sampling:latest \
                 --build-arg=http_proxy=$proxy_url \
                 --build-arg=https_proxy=$proxy_url .
            ;;
        -r|--run)
            sudo docker run -d \
                 -it \
                 --name haq-test \
                 --mount type=bind,source=$SRC_DIR,target=$TARGET_DIR \
                 haq-sampling:latest
            ;;
        -p|--pull)
            git pull
            ;;
        --help|*)
            usage
            ;;
    esac
    shift
done

exit 0
