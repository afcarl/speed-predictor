#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import metrics
import numpy as np

from mach.model import create_optical_flow_model, MobileNetSlim
from mach.util import isAWS, upload_s3, stop_instance, full_path
from mach.data import create_mobilenet_generators

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


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_epochs', 1, 'Number of training examples.')
flags.DEFINE_integer('model_file', None, 'If defined loaded a saved model and continue training.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_string('folder', 'optical_flow_2_augmented_5', 'The folder inside the data folder where the images and labels are.')
flags.DEFINE_float('alpha', 0.75, 'The alpha param for MobileNet.')
flags.DEFINE_boolean('debug', False, 'If this is for debugging the model/training process or not.')
flags.DEFINE_integer('verbose', 0, 'Whether to use verbose logging when constructing the data object.')
flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')
flags.DEFINE_float('min_delta', 0.005, 'Early stopping minimum change value.')
flags.DEFINE_integer('patience', 20, 'Early stopping epochs patience to wait before stopping.')

def main(_):
	config = Config(FLAGS.folder, FLAGS.max_epochs, FLAGS.batch_size, FLAGS.min_delta, FLAGS.patience, FLAGS.alpha)
	print("Training model named", config.model_name())

	folder_path = full_path("data/{}".format(FLAGS.folder))

	train, valid, input_shape = create_mobilenet_generators(folder_path, FLAGS.batch_size, FLAGS.debug)
	train_steps_per_epoch, train_generator = train
	valid_steps_per_epoch, valid_generator = valid

	print("Creating model with input", input_shape)

	if FLAGS.model_file:
		model = load_model(full_path(FLAGS.model_file))
	else:
		# model = create_optical_flow_model(input_shape, FLAGS.alpha)
		model = MobileNetSlim(input_shape, FLAGS.alpha)

	if FLAGS.debug:
		print(model.summary())
		callbacks = None
	else:
		callbacks = [
			ModelCheckpoint(config.model_checkpoint(), verbose=FLAGS.verbose, save_best_only=True),
			CSVLogger(config.csv_log_file()),
			EarlyStopping(monitor='val_loss', min_delta=FLAGS.min_delta, patience=FLAGS.patience, verbose=1)
		]

	print("Compiling model.")
	model.compile(
		optimizer=Adam(lr=0.001),
		loss={'mobilenet_slim_output': 'mean_squared_error'},
		metrics=['mean_absolute_error'])

	print("Starting model train process.")
	model.fit_generator(train_generator,
		train_steps_per_epoch,
		epochs=FLAGS.max_epochs,
		verbose=FLAGS.verbose,
		callbacks=callbacks,
		validation_data=valid_generator,
		validation_steps=valid_steps_per_epoch)

	print("Finished training model.")

	if isAWS() and FLAGS.debug == False:
		upload_s3(config.model_checkpoint())
		upload_s3(config.csv_log_file())

	if isAWS() and FLAGS.stop:
		stop_instance()


if __name__ == '__main__':
	tf.app.run()
