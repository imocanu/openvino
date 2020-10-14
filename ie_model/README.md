# IE_MODEL

## Steps to run model optimizer for TF model :
e.g.
/opt/intel/openvino_2020.4.287/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir ../tf_model/saved_model_TF --input_shape [1,5,32] --output_dir saved_model/fp32 --data_type FP32

> **NOTE**: TF model was configured to accept input_shape [1,5,32] 

## run_ie.py
e.g.
python3 run_ie.py -d CPU -m saved_model/fp32/saved_model.xml

> **NOTE**: TF model is running only for CPU device (FP32)
> **NOTE**: TCN layer is not compatible with MYRIAD device 
> **NOTE**: Output contains : predicted price , mean absolute error for TF model and plot ( with buy/sell signals )
