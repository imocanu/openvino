�	��(_в@��(_в@!��(_в@	�����?�����?!�����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��(_в@�:M��?A}�F�&@YP8��L��?*	E���ԠR@2F
Iterator::Modelᖏ����?!�)��y�F@)� w�(�?1�~UZ>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlZ)r��?!澴�ܚ9@)�8b->�?1)����4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����c>�?!�]s��I5@)�f��I}�?1�
 �0@:Preprocessing2U
Iterator::Model::ParallelMapV2�PoF͇?!�{w�1/@)�PoF͇?1�{w�1/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipW[��잤?!9�7�K@)�}��Aq?1�Xx{��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�9��!l?!􆟯2o@)�9��!l?1􆟯2o@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�����k?!�ݕ��W@)�����k?1�ݕ��W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapIZ��c�?!U�l@��6@)���מYR?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�����?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:M��?�:M��?!�:M��?      ��!       "      ��!       *      ��!       2	}�F�&@}�F�&@!}�F�&@:      ��!       B      ��!       J	P8��L��?P8��L��?!P8��L��?R      ��!       Z	P8��L��?P8��L��?!P8��L��?JCPU_ONLYY�����?b Y      Y@q����Ch�?"�
both�Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 