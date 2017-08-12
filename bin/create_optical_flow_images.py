#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import time

from mach.util import full_path
from mach.data import create_optical_flow_data

flags = tf.app.flags
FLAGS = flags.FLAGS

class Config():
	def __init__(self, num_images, num_augmentations):
		self.num_images = num_images
		self.num_augmentations = num_augmentations

	def folder_path(self):
		return full_path("data/optical_flow_{}_augmented_{}".format(self.num_images, self.num_augmentations))

flags.DEFINE_integer('num_images', 2, 'The number of images in an optical flow batch')
flags.DEFINE_boolean('num_augmentations', 5, 'The number of times a set of image should be augmented.')
flags.DEFINE_boolean('is_test', False, 'If this is for the test video or not.')

def main(_):
	if FLAGS.is_test:
		assert(True == False)

	config = Config(FLAGS.num_images, FLAGS.num_augmentations)

	if os.path.exists(config.folder_path()) == False:
		os.makedirs(config.folder_path())
		os.makedirs(config.folder_path() + "/train")
		os.makedirs(config.folder_path() + "/valid")

	start = time.time()
	print("About to create optical flow images.")
	create_optical_flow_data(FLAGS.num_images, FLAGS.num_augmentations, config.folder_path())
	run_time = time.time() - start
	print("Finished creating opfical flow images. Took {0:.2f} seconds.".format(run_time))

if __name__ == '__main__':
	tf.app.run()
