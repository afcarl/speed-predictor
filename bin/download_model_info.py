#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf

from mach.util import download_s3

flags = tf.app.flags
FLAGS = flags.FLAGS

class Config():
	def __init__(self, folder, max_epochs, batch_size, min_delta, patience, alpha):
		self.folder = folder
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.min_delta = int(min_delta*1000)
		self.patience = patience
		self.alpha = int(alpha*100)

	def model_name(self):
		return "{}_{}_{}_{}_{}_{}".format(self.folder, self.max_epochs, self.batch_size, self.min_delta, self.patience, self.alpha)

	def model_checkpoint(self):
		return "{}.ckpt".format(self.model_name())

	def csv_log_file(self):
		return "{}.csv".format(self.model_name())	

flags.DEFINE_integer('max_epochs', 1, 'Number of training examples.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_string('folder', 'optical_flow_2_augmented_5', 'The folder inside the data folder where the images and labels are.')
flags.DEFINE_float('alpha', 0.75, 'The alpha param for MobileNet.')
flags.DEFINE_float('min_delta', 0.005, 'Early stopping minimum change value.')
flags.DEFINE_integer('patience', 20, 'Early stopping epochs patience to wait before stopping.')

def main(_):
	config = Config(FLAGS.folder, FLAGS.max_epochs, FLAGS.batch_size, FLAGS.min_delta, FLAGS.patience, FLAGS.alpha)
	print("Downloading model info for model {}.".format(config.model_name()))

	download_s3(config.model_checkpoint())
	download_s3(config.csv_log_file())

	print("Done downloading model info")

if __name__ == '__main__':
	tf.app.run()
