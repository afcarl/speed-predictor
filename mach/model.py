#!/bin/python 

from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet, _depthwise_conv_block, _conv_block
from keras import backend as K

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
	x = Conv2D(256, (1,1), padding='same')(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(128)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(dropout)(x)
	x = Dense(32)(x)
	output = Dense(1, name='mobilenet_plus_output')(x)

	model = Model(inputs=encoder_inputs, outputs=output, name='optical_flow_model_mobilenet')
	return model

def create_optical_flow_model(input_shape, alpha):
	input = Input(shape=input_shape)
	encoder = MobileNet(input_tensor=input, alpha=alpha, include_top=False, pooling='avg')

	net = Dropout(0.5)(encoder.output)
	speed = Dense(1, name='speed')(net)

	model = Model(inputs=input, outputs=speed, name='optical_flow_model_mobilenet')
	return model

def MobileNetSlim(input_shape, alpha, depth_multiplier=1, output_classes=1, dropout=0.7):
	input = Input(shape=input_shape)

	x = _conv_block(input, 32, alpha, strides=(2, 2))
	x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
	x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
	x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

	x = GlobalAveragePooling2D()(x)
	x = Dense(128)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Dropout(dropout)(x)
	output = Dense(output_classes, name='mobilenet_slim_output')(x)

	model = Model(inputs=input, outputs=output, name='optical_flow_model_mobilenet')
	return model
