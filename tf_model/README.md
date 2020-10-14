# TF_MODEL scripts

## generateModel.py 
- script is used to load CSV file from data folder , split data for train/test and create/compile TF generateModel
> **NOTE**: Model is saved to saved_model_TF folder

> **NOTE**: Checkpointers are saved to results

> **NOTE**: Tensorboard data is saved to logs folder

## modelBitcoin.py  
- script provides functions to load CSV file, create/compile model, plot generation, signal prediction and data prediction

## params.py

## Requirements :
- docker image : docker pull openvino/ubuntu18_runtime
* optional command to run docker image and enable X : xhost + && docker run -it -u 0 --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --rm openvino/ubuntu18_runtime

- install Keras TCN project : pip3 install keras-tcn
- install BeautifulSoup : pip3 install beautifulsoup4
