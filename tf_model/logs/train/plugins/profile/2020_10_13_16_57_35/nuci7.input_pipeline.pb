	pUj� @pUj� @!pUj� @	U��t�2�?U��t�2�?!U��t�2�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$pUj� @s�����@A[��vN��?Y��<�!7�?*	���(\�Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat^���?!a���P?@)�)�J=�?1�R��t�8@:Preprocessing2F
Iterator::ModelcD�в�?!��ŕ�CE@)><K�P�?1	&�7@:Preprocessing2U
Iterator::Model::ParallelMapV2�M�#~Ŋ?!����b�2@)�M�#~Ŋ?1����b�2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipŪA�۽�?!OS:j;�L@)���?1O��+��%@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap\�=��?!.XdM~.@)3�&c`}?1}��WY$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���O=r?!�a94�D@)���O=r?1�a94�D@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���QIm?!`�Z\I@)���QIm?1`�Z\I@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9V��t�2�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	s�����@s�����@!s�����@      ��!       "      ��!       *      ��!       2	[��vN��?[��vN��?![��vN��?:      ��!       B      ��!       J	��<�!7�?��<�!7�?!��<�!7�?R      ��!       Z	��<�!7�?��<�!7�?!��<�!7�?JCPU_ONLYYV��t�2�?b 