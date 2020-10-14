# openvino
Bitcoin price prediction using Deep Learning ( Tensorflow ) and OpenVINO toolkit

## Project details:
- Project is using TCN layers ( Non-causal architecture ) 
- TCN implementation was done by Philippe Rémy : https://github.com/philipperemy/keras-tcn    
- Input is 3D tensor with shape ( batch_size, timesteps, nb_filters )
- Output is a price prediction for next day

## Requirements :
- (Optional) docker image : docker pull openvino/ubuntu18_runtime
- (Optional) command to run docker image and X : xhost + && docker run -it -u 0 --device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' -v /dev/bus/usb:/dev/bus/usb --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --rm openvino/ubuntu18_runtime
- Install Keras TCN and BeautifulSoup projects

## Run project :
- Generate TF model : python3 tf_model/generateModel.py
- Convert TF model to IE : /opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir ../tf_model/saved_model_TF --input_shape [1,5,32] --output_dir saved_model/fp32 --data_type FP32
- Run IE model on CPU device : python3 run_ie.py -d CPU -m saved_model/fp32/saved_model.xml 

