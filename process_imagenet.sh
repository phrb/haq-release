#! /bin/bash

set -e

IFS="_"

for IMG in *.JPEG; do
    echo $IMG
    read -ra DATA <<< "$IMG"
    mkdir -p "${DATA[0]}"
    mv "${DATA[0]}_${DATA[1]}" "${DATA[0]}/${DATA[1]}"
done
