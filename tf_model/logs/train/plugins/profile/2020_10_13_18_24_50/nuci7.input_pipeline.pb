	��3g}�@��3g}�@!��3g}�@	�F�^ @�F�^ @!�F�^ @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��3g}�@�[!���?A[rP�L�?Y��]�p�?*	-��臨N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�c[���?!sx���i@@)�1���?1C8�`�:@:Preprocessing2F
Iterator::Model�S���?!_��YE@)'ݖ�g�?1΋�B�:@:Preprocessing2U
Iterator::Model::ParallelMapV2�)H�?!�Bq�0@)�)H�?1�Bq�0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��v� ݇?!6��2@)/�N[#��?1����+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����-�?!����L@) �C��<n?1�����@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensord�����m?!��B5��@)d�����m?1��B5��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceL��pvki?!����4@)L��pvki?1����4@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�F�^ @>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�[!���?�[!���?!�[!���?      ��!       "      ��!       *      ��!       2	[rP�L�?[rP�L�?![rP�L�?:      ��!       B      ��!       J	��]�p�?��]�p�?!��]�p�?R      ��!       Z	��]�p�?��]�p�?!��]�p�?JCPU_ONLYY�F�^ @b 