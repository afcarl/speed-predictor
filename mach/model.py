#!/bin/python 

from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.applications.mobilenet import MobileNet
from keras import backend as K


def create_optical_flow_model(input_shape, alpha):
	encoder = MobileNet(input_shape=input_shape, alpha=alpha, include_top=False, weights=None, pooling='avg')
	net = Dropout(0.7)(encoder.output)
	speed = Dense(1, name='speed')(net)

	model = Model(inputs=encoder.inputs, outputs=[speed], name='optical_flow_model')
	return model
