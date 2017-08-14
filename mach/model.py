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

def recurrent_net(input_shape, sequence_size, alpha):
	sequence_input = Input(shape=[sequence_size] + list(input_shape))
	encoder = MobileNetSlim(input_shape, alpha)
	
	encoded_image_sequence = TimeDistributed(encoder)(sequence_input)
	encoded_video = LSTM(128)(encoded_image_sequence)

	x = BatchNormalization()(encoded_video)
	x = Dropout(0.5)(x)
	x = Dense(32)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	output = Dense(1, name='speed')(x)

	model = Model(inputs=sequence_input, outputs=output, name='optical_flow_recurrent')
	return model

def create_mobilenet_plus_model(input_shape, num_images, alpha, dropout=0.5):
	encoder_input = Input(shape=input_shape)
	encoder = MobileNet(input_tensor=encoder_input, alpha=alpha, include_top=False, pooling=None, weights=None)
	encoder_model = Model(inputs=encoder_input, outputs=encoder.output, name='mobilenet_shared')

	encoder_outputs = []
	encoder_inputs = []
	for i in range(num_images):
		encoder_input = Input(shape=input_shape, name="frame_{}".format(i))
		encoder_inputs.append(encoder_input)
		encoder_outputs.append(encoder_model(encoder_input))

	x = concatenate(encoder_outputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.7)(x)
	x = Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(dropout)(x)
	x = Dense(32, kernel_regularizer=regularizers.l2(0.01))(x)
	output = Dense(1, name='mobilenet_plus_output')(x)

	model = Model(inputs=encoder_inputs, outputs=output, name='optical_flow_model_mobilenet')
	return model


def create_optical_flow_model(input_shape, num_images, alpha):
	encoder_input = Input(shape=input_shape)
	encoder = MobileNet(input_tensor=encoder_input, alpha=alpha, include_top=False, pooling=None, weights=None)
	encoder_model = Model(inputs=encoder_input, outputs=encoder.output, name='mobilenet_shared')

	for layer in encoder.layers:
		layer.trainable = False

	encoder_outputs = []
	encoder_inputs = []
	for i in range(num_images):
		encoder_input = Input(shape=input_shape, name="frame_{}".format(i))
		encoder_inputs.append(encoder_input)
		encoder_outputs.append(encoder_model(encoder_input))

	optical_model = MobileNetSlim(input_shape, alpha)

	x = concatenate(encoder_outputs + optical_model.outputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.7)(x)
	x = Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
	x = Flatten()(x)
	x = BatchNormalization()(x)
	x = Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(32, kernel_regularizer=regularizers.l2(0.01))(x)
	output = Dense(1, name='speed')(x)

	model = Model(inputs=encoder_inputs + optical_model.inputs, outputs=output, name='optical_flow_model_mobilenet')
	return model

def create_mobilenet_model(input_shape, alpha):
	input = Input(shape=input_shape)
	encoder = MobileNet(input_tensor=input, alpha=alpha, include_top=False, pooling=None)

	for layer in encoder.layers:
		layer.trainable = False

	model = Model(inputs=input, outputs=encoder.output, name='mobilenet_model')
	return model

def create_simple_optical_flow_model(input_shape):
	input = Input(shape=input_shape, name='flow')
	x = Conv2D(24, (5,5), strides=(2,2))(input)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Conv2D(36, (5,5), strides=(2,2))(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Conv2D(48, (5,5), strides=(2,2))(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Conv2D(64, (3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Conv2D(128, (3,3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.7)(x)
	x = Dense(128)(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Dropout(0.5)(x)
	x = Dense(64)(x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = Dense(1, name='speed')(x)
	model = Model(inputs=input, outputs=x, name='mobilenet_model')
	return model

def MobileNetSlim(input_shape, alpha, depth_multiplier=1, output_classes=1, dropout=0.5):
	input = Input(shape=input_shape, name='flow')

	x = _conv_block(input, 32, alpha, strides=(2, 2))
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
	# x = Dropout(0.7)(x)

	x = GlobalAveragePooling2D()(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout)(x)
	model = Model(inputs=input, outputs=x, name='optical_flow_encoder')
	return model
	x = Dense(32)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	output = Dense(1, name='speed')(x)

	model = Model(inputs=input, outputs=output, name='optical_flow_encoder')
	return model
