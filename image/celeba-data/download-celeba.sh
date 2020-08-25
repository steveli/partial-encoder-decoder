#!/bin/bash

fileid=0B7EVK8r0v71pZjFTYXZWM3FlRnM
filename=img_align_celeba.zip

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip $filename
rm -f cookie $filename
