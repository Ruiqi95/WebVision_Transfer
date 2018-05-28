#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os, re
import webV_process as process
import random
import envvar as envVar
random.seed(10)

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 256
NUM_CLASSES = 500
CLASS_REDICT = dict()

def _filename_to_int(filename):
	return (int(re.findall("/q\d\d\d\d/", filename)[0][2:-1]) - 1)

def _filename_to_int2(filename):
	return (int(re.findall("\d+", filename)[0]) - 1)

def _parse_function(filename, label):
	image_string  = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
	image_float   = tf.cast(image_decoded, tf.float32)
	image_resized = tf.image.resize_images(image_float, [224, 224])
	#image_crop    = tf.random_crop(image_float, [224, 224, 3], seed = 10)
	image_normal  = tf.image.per_image_standardization(image_resized)
	label_64	= tf.cast(label, tf.int64)
	
	return image_normal, label_64

def _parse_function_dsn_target(filename):

	image_string  = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels = 3)
	image_float   = tf.cast(image_decoded, tf.float32)
	image_resized = tf.image.resize_images(image_float, [224, 224])
	#image_crop    = tf.random_crop(image_float, [224, 224, 3], seed = 10)
	image_normal  = tf.image.per_image_standardization(image_resized)
	
	return image_normal

def trainFile():
	print("Importing train data...")
	if(not envVar.DSN):
		raw_filename = process.getTrain100()

		# re-label
		count = 0
		for i in raw_filename:
			if (i[0] in CLASS_REDICT):
				raise ValueError("Duplicated Classes!!!")
			CLASS_REDICT.update({i[0]:count})
			i[0] = count
			count = count + 1
		flat_filename = [[i, j[0]] for j in raw_filename for i in j[2]]

		return flat_filename

	else:
		raw_filename, raw_filename_target = process.getTrain_dsn()

		# re-label
		count = 0
		for i in raw_filename:
			if (i[0] in CLASS_REDICT):
				raise ValueError("Duplicated Classes!!!")
			CLASS_REDICT.update({i[0]:count})
			i[0] = count
			count = count + 1
		flat_filename = [[i, j[0]] for j in raw_filename for i in j[2]]
		flat_filename_target = [[i, j[0]] for j in raw_filename_target for i in j[2]]

		return flat_filename, flat_filename_target

def valFile():
	print("Importing validation data...")
	raw_filename = process.getVal100()
	# re-label
	for i in raw_filename:
		i[0] = CLASS_REDICT[i[0]]
	flat_filename = [[i, j[0]] for j in raw_filename for i in j[2]]

	return flat_filename

def inputs(eval_data, batch_size):
	if not eval_data:
		dir_and_labels = trainFile()
		random.shuffle(dir_and_labels)
		filenames = [i[0] for i in dir_and_labels]
		labels  = [i[1] for i in dir_and_labels]
		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.map(_parse_function)
		dataset = dataset.shuffle(buffer_size=8192, seed=10, reshuffle_each_iteration=True)
		dataset = dataset.batch(batch_size)
		dataset = dataset.repeat()
	else:
		dir_and_labels = valFile()
		#filenames = [os.path.join(test_dir, i) for i in os.listdir(test_dir)]
		#labels    = [_filename_to_int2(i)/50 for i in filenames]
		filenames = [i[0] for i in dir_and_labels]
		labels  = [i[1] for i in dir_and_labels]
		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(int(len(filenames)/5))
  
	return dataset, len(filenames)

def inputs_dsn(eval_data, batch_size):
	if not eval_data:
		dir_and_labels_source, dir_and_labels_target  = trainFile()
		random.shuffle(dir_and_labels_source)
		random.shuffle(dir_and_labels_target)
		filenames_source = [i[0] for i in dir_and_labels_source]
		labels_source  = [i[1] for i in dir_and_labels_source]
		dataset_source = tf.data.Dataset.from_tensor_slices((filenames_source, labels_source))
		dataset_source = dataset_source.map(_parse_function)
		#dataset_source = dataset_source.shuffle(buffer_size=8192, seed=10, reshuffle_each_iteration=True)
		dataset_source = dataset_source.batch(batch_size)
		dataset_source = dataset_source.repeat()

		filenames_target = [i[0] for i in dir_and_labels_target]
		dataset_target = tf.data.Dataset.from_tensor_slices((filenames_target))
		dataset_target = dataset_target.map(_parse_function_dsn_target)
		#dataset_target = dataset_target.shuffle(buffer_size=8192, seed=10, reshuffle_each_iteration=True)
		dataset_target = dataset_target.batch(batch_size)
		dataset_target = dataset_target.repeat()

		dataset = [dataset_source, dataset_target]
		filenames = filenames_source

	else:
		dir_and_labels = valFile()
		filenames = [i[0] for i in dir_and_labels]
		labels  = [i[1] for i in dir_and_labels]
		dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
		dataset = dataset.map(_parse_function)
		dataset = dataset.batch(32)
  
	return dataset, len(filenames)

	
