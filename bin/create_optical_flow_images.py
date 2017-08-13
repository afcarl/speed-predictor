#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import time

from mach.util import full_path
from mach.data import create_optical_flow_data, create_optical_flow_data_recurrent

flags = tf.app.flags
FLAGS = flags.FLAGS

class Config():
	def __init__(self, num_images, num_augmentations, is_recurrent):
		self.num_images = num_images
		self.num_augmentations = num_augmentations
		self.is_recurrent = is_recurrent

	def folder_path(self):
		if self.is_recurrent:
			return full_path("data/optical_flow_recurrent")
		else:
			return full_path("data/optical_flow_{}_augmented_{}".format(self.num_images, self.num_augmentations))

flags.DEFINE_integer('num_images', 2, 'The number of images in an optical flow batch')
flags.DEFINE_integer('num_augmentations', 5, 'The number of times a set of image should be augmented.')
flags.DEFINE_boolean('is_test', False, 'If this is for the test video or not.')
flags.DEFINE_boolean('is_recurrent', True, 'If this is processing for a recurrent model or not.')

def main(_):
	if FLAGS.is_test:
		assert(True == False)

	config = Config(FLAGS.num_images, FLAGS.num_augmentations, FLAGS.is_recurrent)

	if os.path.exists(config.folder_path()) == False:
		os.makedirs(config.folder_path())
		if FLAGS.is_recurrent == False:
			os.makedirs(config.folder_path() + "/train")
			os.makedirs(config.folder_path() + "/valid")

	start = time.time()
	print("About to create optical flow images.")
	if FLAGS.is_recurrent:
		create_optical_flow_data_recurrent(config.folder_path())
	else:
		create_optical_flow_data(FLAGS.num_images, FLAGS.num_augmentations, config.folder_path())
	run_time = time.time() - start
	print("Finished creating opfical flow images. Took {0:.2f} seconds.".format(run_time))

if __name__ == '__main__':
	tf.app.run()
