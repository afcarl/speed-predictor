#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
import numpy as np
from keras.models import load_model

from mach.util import full_path
from mach.data import create_mobilenet_full_generator, raw_train_data

class Config():
	def __init__(self, folder, max_epochs, batch_size, min_delta, patience, alpha=0.75, is_recurrent=False):
		self.folder = folder
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.min_delta = int(min_delta*1000)
		self.patience = patience
		self.alpha = int(alpha*100)
		if is_recurrent:
			self.is_recurrent = "recurrent"
		else:
			self.is_recurrent = "non_recurrent"

	def model_name(self):
		return "{}_{}_{}_{}_{}_{}_{}".format(self.folder, self.max_epochs, self.batch_size, self.min_delta, self.patience, self.alpha, self.is_recurrent)

	def model_checkpoint(self):
		return "{}.ckpt".format(self.model_name())


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_epochs', 200, 'Number of training examples.')
flags.DEFINE_integer('batch_size', 256, 'The batch size for the generator')
flags.DEFINE_string('folder', 'optical_flow_3_augmented_1', 'The folder inside the data folder where the images and labels are.')
flags.DEFINE_boolean('debug', False, 'If this is for debugging the model/training process or not.')
flags.DEFINE_float('min_delta', 0.01, 'Early stopping minimum change value.')
flags.DEFINE_integer('patience', 30, 'Early stopping epochs patience to wait before stopping.')
flags.DEFINE_integer('num_images', 3, 'The number of images used to make the opticlal flow.')
flags.DEFINE_string('video_file', 'test', 'Which video to predict from.')

def main(_):
	config = Config(FLAGS.folder, FLAGS.max_epochs, FLAGS.batch_size, FLAGS.min_delta, FLAGS.patience)
	print("Training model named", config.model_name())

	if FLAGS.video_file == 'train':
		video_folder = 'train_full'
	elif FLAGS.video_file == 'test':
		video_folder = 'test'
	else:
		raise ValueError("Unexpected video file {}".format(FLAGS.video_file))

	folder_path = full_path("data/{}/{}".format(FLAGS.folder, video_folder))

	steps, generator = create_mobilenet_full_generator(folder_path, FLAGS.debug)

	model = load_model(config.model_checkpoint())

	print("Compiling model.")
	model.compile('sgd', 'mse')

	print("Starting model prediction process.")

	predictions = model.predict_generator(generator, steps, verbose=1)

	# b/c the first `num_images` predictions are skipped, add these back based on the first value
	predictions = np.concatenate((np.ones(FLAGS.num_images-1)*predictions[0][0], predictions.reshape(-1)))

	if FLAGS.video_file == 'train':
		_, labels = raw_train_data()
		labels = labels[:len(predictions)]
		print("MSE:", np.mean(np.square(predictions - labels)))

	file = "results_{}.txt".format(FLAGS.video_file)

	print("Saving predictions to {}".format(file))

	np.savetxt(full_path(file), predictions, fmt='%.5f')

if __name__ == '__main__':
	tf.app.run()
