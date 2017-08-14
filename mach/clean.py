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

class BatchGenerator(object):
	def __init__(self, sequence, seq_len, batch_size, processor):
		self.sequence = sequence
		self.seq_len = seq_len
		self.batch_size = batch_size
		chunk_size = 1 + (len(sequence) - 1) / batch_size
		self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]
        
	def next(self):
		while True:
			output = []
			for i in range(self.batch_size):
				idx = self.indices[i]
				left_pad = self.sequence[idx - LEFT_CONTEXT:idx]

				if len(left_pad) < LEFT_CONTEXT:
				    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
				assert len(left_pad) == LEFT_CONTEXT

				leftover = len(self.sequence) - idx
				if leftover >= self.seq_len:
				    result = self.sequence[idx:idx + self.seq_len]
				else:
				    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
				assert len(result) == self.seq_len
				
				self.indices[i] = (idx + self.seq_len) % len(self.sequence)

				images, targets = zip(*result)
				images_left_pad, _ = zip(*left_pad)
				output.append((np.stack(images_left_pad + images), np.stack(targets)))

			output = zip(*output)
			output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
			output[1] = np.stack(output[1]) # batch_size x seq_len x OUTPUT_DIM
			return output


def adam(lr):
	return Adam(lr, clipnorm=15.)






