#!/bin/python 

from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.applications.mobilenet import MobileNet
from keras import backend as K


def create_optical_flow_model(input_shape, alpha):
	input = Input(shape=input_shape)
	encoder = MobileNet(input_tensor=input, alpha=alpha, include_top=False, pooling='avg')
	net = Dropout(0.5)(encoder.output)
	speed = Dense(1, name='speed')(net)

	model = Model(inputs=input, outputs=speed, name='optical_flow_model_mobilenet')
	return model
