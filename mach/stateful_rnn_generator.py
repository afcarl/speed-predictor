#!/bin/python 

from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Activation, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block, _conv_block
from keras import backend as K
from keras.layers.recurrent import LSTM

from keras.optimizers import Adam

import numpy as np

SEQ_LEN = 10 
BATCH_SIZE = 4 
LEFT_CONTEXT = 5
# These are the input image parameters.
HEIGHT = 160
WIDTH = 320
CHANNELS = 3 # RGB
# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32
OUTPUT_DIM = 1

def create_crossvalidate_indicies(sequence_size, batch_size, valid_size=0.3):
	train_chunk_size = int(sequence_size * (1.-valid_size) // batch_size)
	valid_chunk_size = int(sequence_size * valid_size // batch_size)
	train_idxs = []
	valid_idxs = []
	current_idx = 0
	while len(train_idxs) < batch_size or len(valid_idxs) < batch_size:
		if len(train_idxs) < batch_size and len(valid_idxs) < batch_size:
			if np.random.uniform() < 0.5:
				train_idxs.append(current_idx)
				current_idx += train_chunk_size
			else:
				valid_idxs.append(current_idx)
				current_idx += valid_chunk_size
		elif len(valid_idxs) < batch_size:
			valid_idxs.append(current_idx)
			current_idx += valid_chunk_size
		else:
			train_idxs.append(current_idx)
			current_idx += train_chunk_size

	return train_idxs, valid_idxs, train_chunk_size, valid_chunk_size

class CrossValidationBatchSequenceGenerator(object):
	def __init__(self, key, idxs, sequence, seq_len, batch_size, processor, parent):
		self.idxs = idxs
		self.sequence = sequence
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.processor = processor
		self.parent = parent
		self.key = key
		
	def __next__(self):
		return self.next()

	def next(self):
		output = []
		for i in range(self.batch_size):
			idx = self.idxs[i]
			left_pad = self.sequence[idx - LEFT_CONTEXT:idx]

			if len(left_pad) < LEFT_CONTEXT:
			    left_pad = np.array([self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad)
			assert len(left_pad) == LEFT_CONTEXT

			leftover = len(self.sequence) - idx
			if leftover >= self.seq_len:
			    result = self.sequence[idx:idx + self.seq_len]
			else:
			    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
			assert len(result) == self.seq_len
			
			self.idxs[i] = (idx + self.seq_len) % len(self.sequence)

			image_files, targets = list(zip(*result))
			image_files_left_pad, _ = list(zip(*left_pad))
			images = list(map(self.processor, np.stack(image_files_left_pad + image_files)))
			output.append((images, targets[-1]))

		self.parent.increment(self.key)
		return output

class CrossValidationBatchSequenceGeneratorParent(object):
	def __init__(self, image_files, labels, seq_len, batch_size, processor):
		self.sequence = list(zip(image_files, labels))
		self.seq_len = seq_len
		self.batch_size = batch_size
		self.current_valid_epoch_step = 0
		self.new_idxs()
		self.train_generator = CrossValidationBatchSequenceGenerator('train', 
			self.train_idxs, self.sequence, self.seq_len, self.batch_size, processor, self)
		self.valid_generator = CrossValidationBatchSequenceGenerator('valid', 
			self.valid_idxs, self.sequence, self.seq_len, self.batch_size, processor, self)

	def increment(self, key):
		if key == 'valid':
			self.current_valid_epoch_step += 1
			if self.current_valid_epoch_step >= self.valid_epoch_steps:
				self.reset_idxs()

	def new_idxs(self):
		train_idxs, valid_idxs, train_chunk_size, valid_chunk_size = create_crossvalidate_indicies(len(self.sequence), self.batch_size)
		self.train_idxs = train_idxs
		self.valid_idxs = valid_idxs
		self.train_epoch_steps = train_chunk_size // self.seq_len
		self.valid_epoch_steps = valid_chunk_size // self.seq_len

	def reset_idxs(self):
		print("resetting_index")
		self.new_idxs()
		self.current_valid_epoch_step = 0
		self.train_generator.idxs = self.train_idxs
		self.valid_generator.idxs = self.valid_idxs


def adam(lr):
	return Adam(lr, clipnorm=15.)






